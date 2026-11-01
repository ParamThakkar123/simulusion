// Robust Mermaid initialization: wait for mermaid to load, then render only
// unprocessed Mermaid source blocks. Mermaid replaces each rendered block with
// SVG and marks it as data-processed; re-rendering those SVG containers causes
// Mermaid to parse SVG text as diagram source and show "Syntax error in text".
(() => {
  const MAX_WAIT = 5000;
  const INTERVAL = 100;
  let initialized = false;
  let renderQueued = false;

  function normalizeMermaidText(txt) {
    return (txt || '')
      .replace(/[\u201C\u201D]/g, '"')
      .replace(/[\u2018\u2019\u201B]/g, "'")
      .replace(/[\u2013\u2014\u2012]/g, '-')
      .replace(/[-\u2013\u2014]+>/g, '-->')
      .replace(/\u2192/g, '->')
      .replace(/\u2212/g, '-')
      .trim();
  }

  function convertScriptBlocks() {
    document.querySelectorAll('script[type="text/vnd.mermaid"]').forEach((script) => {
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = normalizeMermaidText(script.textContent || script.innerText || '');
      script.parentNode.replaceChild(div, script);
    });
  }

  function normalizeUnprocessedDivs() {
    document.querySelectorAll('.mermaid:not([data-processed="true"])').forEach((node) => {
      if (!node.querySelector('svg')) {
        node.textContent = normalizeMermaidText(node.textContent || node.innerText || '');
      }
    });
  }

  function unprocessedNodes() {
    return Array.from(document.querySelectorAll('.mermaid:not([data-processed="true"])'))
      .filter((node) => !node.querySelector('svg') && normalizeMermaidText(node.textContent || node.innerText || ''));
  }

  function renderMermaid() {
    if (!window.mermaid || !initialized) return;
    convertScriptBlocks();
    normalizeUnprocessedDivs();

    const nodes = unprocessedNodes();
    if (!nodes.length) return;

    try {
      let result;
      if (typeof window.mermaid.run === 'function') {
        result = window.mermaid.run({ nodes });
      } else if (typeof window.mermaid.init === 'function') {
        result = window.mermaid.init(undefined, nodes);
      }
      if (result && typeof result.catch === 'function') {
        result.catch((e) => console.warn('Mermaid render failed', e));
      }
    } catch (e) {
      console.warn('Mermaid render failed', e);
    }
  }

  function queueRender(delay = 50) {
    if (renderQueued) return;
    renderQueued = true;
    setTimeout(() => {
      renderQueued = false;
      renderMermaid();
    }, delay);
  }

  function tryInit() {
    if (!window.mermaid) return false;
    if (!initialized) {
      try {
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        const dark = document.documentElement.dataset.theme === 'dark' || prefersDark;
        window.mermaid.initialize({
          startOnLoad: false,
          securityLevel: 'loose',
          theme: dark ? 'dark' : 'default',
        });
        initialized = true;
      } catch (e) {
        console.warn('Mermaid initialize failed', e);
        return false;
      }
    }

    if (window.MathJax && window.MathJax.startup && typeof window.MathJax.startup.typesetPromise === 'function') {
      window.MathJax.startup.typesetPromise()
        .then(() => queueRender(0))
        .catch(() => queueRender(100));
    } else {
      queueRender(50);
    }
    return true;
  }

  // Poll until mermaid is available or timeout
  let waited = 0;
  const poll = setInterval(() => {
    if (tryInit() || waited > MAX_WAIT) {
      clearInterval(poll);
    }
    waited += INTERVAL;
  }, INTERVAL);

  // Watch for new mermaid blocks added dynamically
  const observer = new MutationObserver(() => {
    if (!window.mermaid || !initialized) return;
    if (unprocessedNodes().length || document.querySelector('script[type="text/vnd.mermaid"]')) {
      queueRender(100);
    }
  });

  function startObserving() {
    const targetNode = document.body || document.documentElement;
    if (targetNode) {
      observer.observe(targetNode, { childList: true, subtree: true });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startObserving, { once: true });
  } else {
    startObserving();
  }
})();
