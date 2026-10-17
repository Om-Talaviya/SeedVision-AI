/* ==========================================================================
   SeedVision AI — static showcase behaviour
   Pure vanilla JS, no dependencies, no network calls, no bundler.
   All analysis output in the demo is a deterministic pixel-statistics
   heuristic and is explicitly labelled "simulated" in the UI.
   ========================================================================== */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ------------------------------------------------------------------ nav */
  function initNav() {
    var toggle = document.getElementById("navToggle");
    var nav = document.getElementById("primaryNav");
    if (!toggle || !nav) return;

    function setOpen(open) {
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      nav.classList.toggle("open", open);
    }

    toggle.addEventListener("click", function () {
      setOpen(toggle.getAttribute("aria-expanded") !== "true");
    });

    nav.addEventListener("click", function (event) {
      if (event.target.closest("a")) setOpen(false);
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && toggle.getAttribute("aria-expanded") === "true") {
        setOpen(false);
        toggle.focus();
      }
    });

    window.addEventListener("resize", function () {
      if (window.innerWidth > 760) setOpen(false);
    });
  }

  /* --------------------------------------------------------------- reveal */
  function initReveal() {
    var items = Array.prototype.slice.call(document.querySelectorAll(".reveal"));
    if (!items.length) return;

    if (reduceMotion || !("IntersectionObserver" in window)) {
      items.forEach(function (el) { el.classList.add("is-visible"); });
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        var el = entry.target;
        var siblings = Array.prototype.slice.call(el.parentNode.children);
        var index = siblings.indexOf(el);
        var delay = Math.min(index, 6) * 70;
        el.style.transitionDelay = delay + "ms";
        el.classList.add("is-visible");
        observer.unobserve(el);
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.12 });

    items.forEach(function (el) { observer.observe(el); });
  }

  /* ------------------------------------------------------------- lightbox */
  function initLightbox() {
    var dialog = document.getElementById("lightbox");
    var image = document.getElementById("lightboxImg");
    var caption = document.getElementById("lightboxTitle");
    var closeBtn = document.getElementById("lightboxClose");
    var triggers = Array.prototype.slice.call(document.querySelectorAll(".shot[data-full]"));
    if (!dialog || !image || !triggers.length) return;

    var lastFocused = null;

    function open(trigger) {
      lastFocused = trigger;
      image.src = trigger.getAttribute("data-full");
      image.alt = trigger.getAttribute("data-title") || "Interface screenshot";
      caption.textContent = trigger.getAttribute("data-title") || "";
      dialog.hidden = false;
      document.body.style.overflow = "hidden";
      if (closeBtn) closeBtn.focus();
    }

    function close() {
      dialog.hidden = true;
      image.removeAttribute("src");
      document.body.style.overflow = "";
      if (lastFocused) lastFocused.focus();
    }

    triggers.forEach(function (trigger) {
      trigger.addEventListener("click", function () { open(trigger); });
    });

    if (closeBtn) closeBtn.addEventListener("click", close);

    dialog.addEventListener("click", function (event) {
      if (event.target === dialog) close();
    });

    dialog.addEventListener("keydown", function (event) {
      if (event.key === "Escape") { close(); return; }
      if (event.key !== "Tab") return;
      if (!closeBtn) return;

      var focusables = dialog.querySelectorAll("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])");
      var list = Array.prototype.filter.call(focusables, function (el) { return el.offsetParent !== null || el === closeBtn; });
      if (!list.length) return;

      var first = list[0];
      var last = list[list.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    });
  }

  /* ----------------------------------------------------- simulated demo --- */
  var WORK = 256;          /* working resolution for pixel statistics          */
  var MIN_INPUT = 224;     /* pipeline input size (224x224)                    */
  var BRIGHT_LOW = 55, BRIGHT_HIGH = 215;
  var CONTRAST_LOW = 22;
  var SHARP_WARN = 120, SHARP_FAIL = 45;

  var GRADES = {
    healthy:    { label: "sound_healthy",      note: "uniform golden surface" },
    spotted:    { label: "discolored_spotted", note: "dark discoloured regions" },
    cracked:    { label: "broken_cracked",     note: "high-contrast fracture line" },
    shrivelled: { label: "shrivelled_damaged", note: "dull, low-contrast surface" }
  };

  var canvas = document.getElementById("previewCanvas");
  var ctx = canvas ? canvas.getContext("2d", { willReadFrequently: true }) : null;

  var el = {
    empty: document.getElementById("previewEmpty"),
    grade: document.getElementById("kpiClass"),
    quality: document.getElementById("kpiQuality"),
    res: document.getElementById("kpiRes"),
    bright: document.getElementById("kpiBright"),
    findings: document.getElementById("findings")
  };

  /* Deterministic synthetic seed swatches, drawn in-browser. */
  function drawSample(kind) {
    var size = canvas.width;
    ctx.clearRect(0, 0, size, size);

    var base = ctx.createRadialGradient(size * 0.36, size * 0.34, size * 0.06, size * 0.5, size * 0.5, size * 0.62);
    var stops = {
      healthy:    ["#e0c37c", "#b8952f"],
      spotted:    ["#d8bb74", "#9c7625"],
      cracked:    ["#e0c37c", "#a8842f"],
      shrivelled: ["#9c8340", "#66521f"]
    }[kind] || ["#e0c37c", "#b8952f"];

    base.addColorStop(0, stops[0]);
    base.addColorStop(1, stops[1]);
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, size, size);

    if (kind === "spotted") {
      var spots = [[0.28, 0.34, 0.13], [0.62, 0.52, 0.10], [0.46, 0.74, 0.08], [0.74, 0.24, 0.07]];
      ctx.fillStyle = "rgba(58, 38, 12, 0.92)";
      spots.forEach(function (s) {
        ctx.beginPath();
        ctx.arc(s[0] * size, s[1] * size, s[2] * size, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    if (kind === "cracked") {
      ctx.strokeStyle = "rgba(38, 26, 8, 0.95)";
      ctx.lineWidth = Math.max(2, size * 0.014);
      ctx.beginPath();
      ctx.moveTo(size * 0.08, size * 0.86);
      ctx.quadraticCurveTo(size * 0.42, size * 0.52, size * 0.52, size * 0.30);
      ctx.quadraticCurveTo(size * 0.62, size * 0.14, size * 0.92, size * 0.10);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(size * 0.46, size * 0.56);
      ctx.lineTo(size * 0.70, size * 0.62);
      ctx.stroke();
    }

    if (kind === "shrivelled") {
      ctx.globalAlpha = 0.5;
      var wrinkles = [[0.20, 0.30, 0.28, 0.08], [0.52, 0.62, 0.34, 0.07], [0.70, 0.40, 0.24, 0.06]];
      wrinkles.forEach(function (w) {
        ctx.fillStyle = "rgba(74, 58, 26, 0.85)";
        ctx.beginPath();
        ctx.ellipse(w[0] * size, w[1] * size, w[2] * size, w[3] * size, 0.4, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalAlpha = 1;
    }

    if (kind === "healthy") {
      var sheen = ctx.createRadialGradient(size * 0.34, size * 0.30, 0, size * 0.34, size * 0.30, size * 0.34);
      sheen.addColorStop(0, "rgba(255, 246, 214, 0.55)");
      sheen.addColorStop(1, "rgba(255, 246, 214, 0)");
      ctx.fillStyle = sheen;
      ctx.fillRect(0, 0, size, size);
    }
  }

  /* Fit an arbitrary image into the working canvas (object-fit: cover). */
  function drawImageCover(img) {
    var size = canvas.width;
    ctx.clearRect(0, 0, size, size);
    var scale = Math.max(size / img.naturalWidth, size / img.naturalHeight);
    var w = img.naturalWidth * scale;
    var h = img.naturalHeight * scale;
    ctx.drawImage(img, (size - w) / 2, (size - h) / 2, w, h);
  }

  /* Downscale the visible canvas to a fixed working buffer and read pixels. */
  function readWorkingPixels() {
    var off = document.createElement("canvas");
    off.width = WORK;
    off.height = WORK;
    var octx = off.getContext("2d", { willReadFrequently: true });
    octx.drawImage(canvas, 0, 0, WORK, WORK);
    return octx.getImageData(0, 0, WORK, WORK).data;
  }

  function computeStats(pixels) {
    var n = WORK * WORK;
    var gray = new Float32Array(n);
    var sumL = 0;
    var dark = 0;
    var i;

    for (i = 0; i < n; i++) {
      var r = pixels[i * 4];
      var g = pixels[i * 4 + 1];
      var b = pixels[i * 4 + 2];
      var l = 0.299 * r + 0.587 * g + 0.114 * b;
      gray[i] = l;
      sumL += l;
      if (l < 90) dark++;
    }

    var mean = sumL / n;
    var variance = 0;
    for (i = 0; i < n; i++) {
      var d = gray[i] - mean;
      variance += d * d;
    }
    var contrast = Math.sqrt(variance / n);

    /* Laplacian (var 8-neighbour) kernel: sharp edges raise the variance. */
    var sumLap = 0, sumLap2 = 0, count = 0;
    for (var y = 1; y < WORK - 1; y++) {
      for (var x = 1; x < WORK - 1; x++) {
        var idx = y * WORK + x;
        var lap = 8 * gray[idx]
          - gray[idx - 1] - gray[idx + 1]
          - gray[idx - WORK] - gray[idx + WORK]
          - gray[idx - WORK - 1] - gray[idx - WORK + 1]
          - gray[idx + WORK - 1] - gray[idx + WORK + 1];
        sumLap += lap;
        sumLap2 += lap * lap;
        count++;
      }
    }
    var lapMean = sumLap / count;
    var sharpness = sumLap2 / count - lapMean * lapMean;

    return {
      brightness: mean,
      contrast: contrast,
      sharpness: sharpness,
      darkFraction: dark / n
    };
  }

  /* Deterministic placeholder grade from pixel statistics.
     Deliberately generic — it is NOT the trained CNN and cannot be. */
  function suggestGrade(stats) {
    if (stats.sharpness > 900) return "cracked";
    if (stats.darkFraction > 0.09) return "spotted";
    if (stats.brightness < 118 && stats.contrast < 46) return "shrivelled";
    if (stats.contrast > 34 && stats.brightness > 130) return "healthy";
    return "shrivelled";
  }

  function finding(kind, text) {
    var li = document.createElement("li");
    li.className = "finding finding-" + kind;
    li.appendChild(document.createTextNode(text));
    return li;
  }

  function runChecks(stats, width, height) {
    var rows = [];
    var worst = "pass";

    function worse(a, b) {
      var rank = { pass: 0, warn: 1, fail: 2 };
      return rank[a] >= rank[b] ? a : b;
    }

    if (width < MIN_INPUT || height < MIN_INPUT) {
      rows.push(["fail", "Resolution " + width + "×" + height + " px is below the 224×224 pipeline input."]);
      worst = worse(worst, "fail");
    } else {
      rows.push(["pass", "Resolution " + width + "×" + height + " px meets the 224×224 pipeline input."]);
    }

    var ratio = Math.max(width / height, height / width);
    if (ratio > 3) {
      rows.push(["warn", "Aspect ratio " + ratio.toFixed(2) + ":1 is unusually elongated for a single-seed framing."]);
      worst = worse(worst, "warn");
    } else {
      rows.push(["pass", "Aspect ratio " + ratio.toFixed(2) + ":1 is within the single-seed framing range."]);
    }

    if (stats.brightness < BRIGHT_LOW) {
      rows.push(["fail", "Mean brightness " + Math.round(stats.brightness) + "/255 suggests under-exposure."]);
      worst = worse(worst, "fail");
    } else if (stats.brightness > BRIGHT_HIGH) {
      rows.push(["warn", "Mean brightness " + Math.round(stats.brightness) + "/255 risks blown-out highlights."]);
      worst = worse(worst, "warn");
    } else {
      rows.push(["pass", "Mean brightness " + Math.round(stats.brightness) + "/255 is in the usable range."]);
    }

    if (stats.contrast < CONTRAST_LOW) {
      rows.push(["warn", "Low luminance spread (σ=" + stats.contrast.toFixed(1) + ") — surface detail may be washed out."]);
      worst = worse(worst, "warn");
    } else {
      rows.push(["pass", "Luminance spread σ=" + stats.contrast.toFixed(1) + " retains surface detail."]);
    }

    if (stats.sharpness < SHARP_FAIL) {
      rows.push(["fail", "Focus metric " + Math.round(stats.sharpness) + " indicates a heavily blurred frame."]);
      worst = worse(worst, "fail");
    } else if (stats.sharpness < SHARP_WARN) {
      rows.push(["warn", "Focus metric " + Math.round(stats.sharpness) + " is soft — a sharper capture is preferred."]);
      worst = worse(worst, "warn");
    } else {
      rows.push(["pass", "Focus metric " + Math.round(stats.sharpness) + " indicates acceptable edge definition."]);
    }

    return { rows: rows, worst: worst };
  }

  function analyse(width, height) {
    if (!canvas || !ctx || !el.findings) return;

    var stats = computeStats(readWorkingPixels());
    var checks = runChecks(stats, width, height);
    var grade = GRADES[suggestGrade(stats)];

    if (el.grade) el.grade.textContent = grade.label;
    if (el.quality) {
      el.quality.textContent = checks.worst === "fail" ? "Screening failed"
        : checks.worst === "warn" ? "Acceptable with warnings"
        : "Passed screening";
    }
    if (el.res) el.res.textContent = width + " × " + height;
    if (el.bright) el.bright.textContent = Math.round(stats.brightness) + " / 255";

    el.findings.textContent = "";
    checks.rows.forEach(function (row) { el.findings.appendChild(finding(row[0], row[1])); });
    el.findings.appendChild(finding(
      "pass",
      "Simulated grade " + grade.label + " (" + grade.note + ") — illustrative only, not a model prediction."
    ));
    if (el.empty) el.empty.hidden = true;
  }

  function initDemo() {
    if (!canvas || !ctx) return;

    var fileInput = document.getElementById("fileInput");
    var samples = Array.prototype.slice.call(document.querySelectorAll(".sample[data-sample]"));

    function clearPressed() {
      samples.forEach(function (b) { b.setAttribute("aria-pressed", "false"); });
    }

    samples.forEach(function (button) {
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", function () {
        var kind = button.getAttribute("data-sample");
        clearPressed();
        button.setAttribute("aria-pressed", "true");
        if (fileInput) fileInput.value = "";
        drawSample(kind);
        analyse(canvas.width, canvas.height);
      });
    });

    if (fileInput) {
      fileInput.addEventListener("change", function () {
        var file = fileInput.files && fileInput.files[0];
        if (!file) return;
        var url = URL.createObjectURL(file);
        var img = new Image();
        img.onload = function () {
          clearPressed();
          drawImageCover(img);
          URL.revokeObjectURL(url);
          analyse(img.naturalWidth, img.naturalHeight);
        };
        img.onerror = function () {
          URL.revokeObjectURL(url);
          if (el.findings) {
            el.findings.textContent = "";
            el.findings.appendChild(finding("fail", "That file could not be decoded as an image."));
          }
        };
        img.src = url;
      });
    }

    drawSample("healthy");
    if (el.empty) el.empty.hidden = true;
  }

  /* ------------------------------------------------------------------ boot */
  function boot() {
    initNav();
    initReveal();
    initLightbox();
    initDemo();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
