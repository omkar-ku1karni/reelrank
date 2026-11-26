/*
 * Tints numbers inside sentences ("95%", "32,487", "5.0 s") so they stand out naturally,
 * the same rule as Omkar Kulkarni's portfolio. Identifiers like A2A, p95 or SigLIP2 are left alone.
 */
(function () {
  var NUM = /(?<![\w.@/-])(~?[+-]?\d[\d,]*(?:\.\d+)?(?:\/\d[\d,]*)?(?:\s?%|\s?ms\b|\s?s\b|k\b|M\b)?)(?![\w%])/g;
  var SCOPE = '.lede, .prose p, .sub, .stat p, .finding p, .dec .w, .limit p, figure > figcaption, .eng p, .gate li, .chart-card .note, .tbl-cap, footer p';
  document.querySelectorAll(SCOPE).forEach(function (root) {
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    var texts = [];
    while (walker.nextNode()) texts.push(walker.currentNode);
    texts.forEach(function (t) {
      if (!t.nodeValue || t.parentElement.closest('code, .n, svg')) return;
      NUM.lastIndex = 0;
      if (!NUM.test(t.nodeValue)) return;
      var frag = document.createDocumentFragment();
      t.nodeValue.split(NUM).forEach(function (part, i) {
        if (!part) return;
        if (i % 2 === 1) {
          var span = document.createElement('span');
          span.className = 'n';
          span.textContent = part;
          frag.appendChild(span);
        } else {
          frag.appendChild(document.createTextNode(part));
        }
      });
      t.parentNode.replaceChild(frag, t);
    });
  });
})();
