// Sets the product name "Simulusion" like the site's navbar wordmark (Geist
// Mono 500, uppercase, 0.16em tracking; see .brand-word in custom.css)
// wherever it appears in prose. Code snippets (blocks, captions, inline code,
// notebook cells, source views, API signatures), maths, form fields and the
// navbar logo are left alone. Search results arrive later, so new nodes are
// watched.
(function () {
  var WORD = /\bSimulusion\b/;
  var SPLIT = /(\bSimulusion\b)/;
  var SKIP = [
    "pre", "code", "kbd", "samp", "tt", "var", "script", "style", "title",
    "textarea", "input", "select", "option", "svg", "math", "mjx-container", ".MathJax",
    "[class*='highlight']", ".literal", ".literal-block-wrapper", ".code-block-caption",
    ".viewcode-block", ".nbinput", ".nboutput", ".sig", "dt.sig", ".doctest",
    ".brand-word", ".navbar-brand",
  ].join(", ");

  function wrapTextNode(node) {
    var parent = node.parentNode;
    if (!parent || !WORD.test(node.nodeValue)) return;
    if (parent.closest && parent.closest(SKIP)) return;
    var frag = document.createDocumentFragment();
    node.nodeValue.split(SPLIT).forEach(function (part) {
      if (!part) return;
      if (part === "Simulusion") {
        var span = document.createElement("span");
        span.className = "brand-word";
        span.textContent = part;
        frag.appendChild(span);
      } else {
        frag.appendChild(document.createTextNode(part));
      }
    });
    parent.replaceChild(frag, node);
  }

  function walk(root) {
    if (!root || (root.nodeType === 1 && root.closest && root.closest(SKIP))) return;
    if (root.nodeType === 3) return wrapTextNode(root);
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        return WORD.test(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      },
    });
    var found = [];
    while (walker.nextNode()) found.push(walker.currentNode);
    found.forEach(wrapTextNode);
  }

  function start() {
    walk(document.body);
    new MutationObserver(function (records) {
      records.forEach(function (r) {
        r.addedNodes.forEach(walk);
      });
    }).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
