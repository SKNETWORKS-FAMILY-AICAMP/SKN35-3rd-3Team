/* This runs in Streamlit's document, not an iframe. No server round trip. */
(() => {
  const slot = '__lawMatePageNavigation';
  window[slot]?.cleanup();
  const cleanups = [];
  let observer;
  let mounted = false;
  const cleanup = () => {
    observer?.disconnect();
    cleanups.splice(0).forEach(remove => remove());
  };
  window[slot] = {cleanup};

  const mount = () => {
    if (mounted) return;
    const button = document.getElementById('law-back-to-top');
    const main = document.querySelector('[data-testid="stMain"]');
    if (!button || !main) return;
    mounted = true;
    observer?.disconnect();
    const update = () => {
      // Nested chat scroll events never determine the button's visibility.
      const offset = Math.max(main.scrollTop, document.scrollingElement?.scrollTop || 0, window.scrollY || 0);
      button.hidden = offset < 240;
    };
    const scrollToTop = () => {
      const behavior = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
      main.scrollTo({top: 0, behavior});
      document.scrollingElement?.scrollTo({top: 0, behavior});
      window.scrollTo({top: 0, behavior});
      document.querySelector('.st-key-law_header button')?.focus({preventScroll: true});
    };
    document.addEventListener('scroll', update, {capture: true, passive: true});
    window.addEventListener('resize', update, {passive: true});
    button.addEventListener('click', scrollToTop);
    cleanups.push(
      () => document.removeEventListener('scroll', update, {capture: true}),
      () => window.removeEventListener('resize', update),
      () => button.removeEventListener('click', scrollToTop),
    );
    update();
  };
  mount();
  if (!mounted) {
    observer = new MutationObserver(mount);
    observer.observe(document.body, {childList: true, subtree: true});
  }
})();
