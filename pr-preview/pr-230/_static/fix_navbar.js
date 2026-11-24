// Search click opens overlay - run after DOM loads
document.addEventListener('DOMContentLoaded', function() {
  var isOpen = false;
  
  // Create our custom overlay
  var overlay = document.createElement('div');
  overlay.id = 'custom-search-overlay';
  overlay.style.cssText = 'display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);backdrop-filter:blur(8px);z-index:10000;align-items:flex-start;justify-content:center;padding-top:12vh;';
  overlay.innerHTML = '<div style="width:90%;max-width:580px;background:var(--pst-color-background,#fff);border-radius:12px;padding:1.5rem;box-shadow:0 25px 50px rgba(0,0,0,0.4);border:1px solid var(--pst-color-border,#e2e8f0);"><input type="text" placeholder="Search the docs..." style="width:100%;font-size:1.15rem;padding:0.75rem 1rem;background:var(--pst-color-surface,#f8fafc);border:2px solid var(--pst-color-border,#e2e8f0);border-radius:8px;color:var(--pst-color-text-base,#1e293b);outline:none;" autofocus><div style="margin-top:0.75rem;text-align:center;color:var(--pst-color-text-muted,#64748b);font-size:0.8rem;"><kbd style="background:var(--pst-color-surface,#f8fafc);padding:0.2rem 0.5rem;border-radius:4px;font-size:0.7rem;">Ctrl</kbd>+<kbd style="background:var(--pst-color-surface,#f8fafc);padding:0.2rem 0.5rem;border-radius:4px;font-size:0.7rem;">K</kbd> to search</div></div>';
  document.body.appendChild(overlay);
  
  var input = overlay.querySelector('input');

  function simplifyNavbar() {
    var header = document.querySelector('.bd-header');
    if (!header) {
      return;
    }

    // Remove generated top-level page navigation while leaving the brand,
    // search field, and GitHub icon link in place.
    header.querySelectorAll('.navbar-header-items__center, .navbar-nav.bd-navbar-elements').forEach(function(el) {
      el.remove();
    });

    // PyData may inject version controls depending on the installed theme
    // version. Keep only search, the light/dark switch and icon links.
    header.querySelectorAll('.navbar-version, .pst-navbar-icon').forEach(function(el) {
      // The light/dark switch carries .pst-navbar-icon too; keep it.
      if (el.closest('.theme-switch-button, .theme-switcher')) {
        return;
      }
      var link = el.closest('a');
      if (!link || !/github\.com/.test(link.href || '')) {
        el.remove();
      }
    });
  }

  simplifyNavbar();
  function openSearchOverlay() {
    if (!isOpen) {
      isOpen = true;
      overlay.style.display = 'flex';
      input.focus();
    }
  }
  
  // Intercept showModal to use our overlay instead
  var dialogProto = window.HTMLDialogElement && HTMLDialogElement.prototype;
  if (dialogProto && dialogProto.showModal) {
    var origShowModal = dialogProto.showModal;
    dialogProto.showModal = function() {
      if (this.id === 'pst-search-dialog') {
        openSearchOverlay();
        return;
      }
      return origShowModal.call(this);
    };
  }
  
  // Direct click handler on search area
  var bdSearch = document.querySelector('.bd-search');
  if (bdSearch) {
    bdSearch.addEventListener('click', function(e) {
      openSearchOverlay();
    });
    bdSearch.addEventListener('mousedown', function(e) {
      e.preventDefault();
    });
  }
  
  // Ctrl+K
  document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      openSearchOverlay();
    }
    if (e.key === 'Escape' && overlay.style.display === 'flex') {
      overlay.style.display = 'none';
    }
  });
  
  // Close on background click
  overlay.addEventListener('click', function(e) {
    if (e.target === overlay) {
      overlay.style.display = 'none';
    }
  });
});