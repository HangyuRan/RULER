/* Progressive enhancement for the RULER research homepage. */
(() => {
  "use strict";

  function initialize() {
    document.documentElement.classList.add("js");

    document.querySelectorAll(".results-tabs").forEach((tabList) => {
      const tabs = Array.from(tabList.querySelectorAll('[role="tab"]'));
      const panels = tabs.map((tab) =>
        document.getElementById(tab.getAttribute("aria-controls"))
      );
      if (!tabs.length || panels.some((panel) => !panel)) return;

      function selectTab(index, moveFocus = false) {
        tabs.forEach((tab, tabIndex) => {
          const selected = tabIndex === index;
          tab.setAttribute("aria-selected", String(selected));
          tab.tabIndex = selected ? 0 : -1;
          panels[tabIndex].hidden = !selected;
        });
        if (moveFocus) tabs[index].focus();
      }

      tabs.forEach((tab, index) => {
        tab.addEventListener("click", () => selectTab(index));
        tab.addEventListener("keydown", (event) => {
          let nextIndex;
          if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
          if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
          if (event.key === "Home") nextIndex = 0;
          if (event.key === "End") nextIndex = tabs.length - 1;
          if (nextIndex === undefined) return;
          event.preventDefault();
          selectTab(nextIndex, true);
        });
      });

      function revealHashTarget(hash) {
        if (!hash || hash === "#") return false;
        let targetId;
        try {
          targetId = decodeURIComponent(hash.slice(1));
        } catch (_) {
          return false;
        }
        const target = document.getElementById(targetId);
        const index = panels.findIndex((panel) => target && (panel === target || panel.contains(target)));
        if (index < 0) return false;
        selectTab(index);
        return true;
      }

      selectTab(0);
      revealHashTarget(window.location.hash);
      document.querySelectorAll('a[href^="#"]').forEach((link) => {
        link.addEventListener("click", () => revealHashTarget(link.hash));
      });
      window.addEventListener("hashchange", () => revealHashTarget(window.location.hash));
    });

    const dialog = document.getElementById("figure-dialog");
    const dialogImage = document.getElementById("dialog-image");
    const dialogTitle = document.getElementById("dialog-title");
    const dialogCaption = document.getElementById("dialog-caption");
    const dialogSource = document.getElementById("dialog-source");
    const dialogZoom = document.getElementById("dialog-zoom");
    let figureTrigger = null;

    if (dialog && dialogImage) {
      function setActualSize(actualSize) {
        dialogImage.classList.toggle("actual-size", actualSize);
        if (dialogZoom) {
          dialogZoom.setAttribute("aria-pressed", String(actualSize));
          dialogZoom.textContent = actualSize ? "Fit to screen" : "Zoom in";
          dialogZoom.setAttribute("aria-label", actualSize ? "Fit figure to screen" : "View figure at full resolution");
        }
      }

      document.querySelectorAll("button.figure-zoom").forEach((button) => {
        button.addEventListener("click", () => {
          const source = button.dataset.image;
          if (!source) return;
          if (typeof dialog.showModal !== "function") {
            window.open(source, "_blank", "noopener,noreferrer");
            return;
          }

          figureTrigger = button;
          dialogImage.src = source;
          const previewImage = button.querySelector("img");
          const sourceCaption = button.closest("figure")?.querySelector("figcaption");
          const caption = button.dataset.caption || sourceCaption?.textContent.trim() || "";
          dialogImage.alt = previewImage?.alt || button.dataset.title || "Paper figure";
          if (dialogTitle) dialogTitle.textContent = button.dataset.title || "Paper figure";
          if (dialogCaption) {
            dialogCaption.textContent = caption;
            dialogCaption.hidden = !caption;
          }
          if (dialogSource) dialogSource.href = source;
          setActualSize(false);
          dialog.showModal();
          document.body.classList.add("dialog-open");
        });
      });

      dialog.querySelectorAll("[data-dialog-close]").forEach((button) => {
        button.addEventListener("click", () => dialog.close());
      });
      dialog.addEventListener("click", (event) => {
        if (event.target !== dialog) return;
        const bounds = dialog.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right ||
            event.clientY < bounds.top || event.clientY > bounds.bottom) {
          dialog.close();
        }
      });
      dialog.addEventListener("close", () => {
        document.body.classList.remove("dialog-open");
        setActualSize(false);
        if (figureTrigger && figureTrigger.isConnected) {
          figureTrigger.focus({ preventScroll: true });
        }
      });
      if (dialogZoom) {
        dialogZoom.addEventListener("click", () => {
          setActualSize(!dialogImage.classList.contains("actual-size"));
        });
      }
    }

    const navigationLinks = Array.from(document.querySelectorAll('.section-link[href^="#"]'));
    const sectionLinks = navigationLinks.map((link) => ({
      link,
      section: document.getElementById(link.hash.slice(1)),
    })).filter((item) => item.section);

    function markCurrentSection(section) {
      sectionLinks.forEach((item) => {
        if (item.section === section) item.link.setAttribute("aria-current", "location");
        else item.link.removeAttribute("aria-current");
      });
    }

    if (sectionLinks.length) {
      const initialSection = sectionLinks.find((item) => item.link.hash === window.location.hash);
      markCurrentSection((initialSection || sectionLinks[0]).section);
      navigationLinks.forEach((link) => {
        link.addEventListener("click", () => {
          const section = document.getElementById(link.hash.slice(1));
          if (section) markCurrentSection(section);
        });
      });

      if ("IntersectionObserver" in window) {
        const visibleSections = new Set();
        const observer = new IntersectionObserver((entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) visibleSections.add(entry.target);
            else visibleSections.delete(entry.target);
          });
          const visibleSection = sectionLinks.find((item) => visibleSections.has(item.section));
          if (visibleSection) markCurrentSection(visibleSection.section);
        }, { rootMargin: "-15% 0px -65% 0px", threshold: 0 });
        sectionLinks.forEach((item) => observer.observe(item.section));
      }
    }

    const copyButton = document.getElementById("copy-citation");
    const citation = document.getElementById("citation-code");
    const copyStatus = document.getElementById("copy-status");

    if (copyButton && citation) {
      const defaultContent = copyButton.innerHTML;
      let feedbackTimeout;

      function syncCitationState() {
        const content = citation.textContent.split(/\r?\n/)
          .filter((line) => !line.trim().startsWith("%"))
          .join("\n").trim();
        copyButton.disabled = !/^@\w+\s*[{(]/m.test(content);
      }

      function copyWithSelection(text) {
        const previousFocus = document.activeElement;
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.setAttribute("readonly", "");
        textarea.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0";
        document.body.appendChild(textarea);
        textarea.select();
        textarea.setSelectionRange(0, text.length);
        let copied = false;
        try {
          copied = document.execCommand("copy");
        } finally {
          textarea.remove();
          if (previousFocus && typeof previousFocus.focus === "function") {
            previousFocus.focus({ preventScroll: true });
          }
        }
        if (!copied) throw new Error("Clipboard is unavailable.");
      }

      syncCitationState();
      new MutationObserver(syncCitationState).observe(citation, {
        childList: true,
        characterData: true,
        subtree: true,
      });

      copyButton.addEventListener("click", async () => {
        if (copyButton.disabled) return;
        const text = citation.textContent.trim();
        clearTimeout(feedbackTimeout);
        if (copyStatus) copyStatus.textContent = "";
        try {
          if (navigator.clipboard && window.isSecureContext) {
            try {
              await navigator.clipboard.writeText(text);
            } catch (_) {
              copyWithSelection(text);
            }
          } else {
            copyWithSelection(text);
          }
          copyButton.textContent = "Copied!";
          if (copyStatus) copyStatus.textContent = "Citation copied to clipboard.";
          feedbackTimeout = window.setTimeout(() => {
            copyButton.innerHTML = defaultContent;
          }, 2200);
        } catch (_) {
          copyButton.innerHTML = defaultContent;
          if (copyStatus) copyStatus.textContent = "Unable to copy automatically. Select and copy the citation text.";
        }
      });
    }

    const menuToggle = document.getElementById("menu-toggle");
    const primaryNav = document.getElementById("primary-nav");
    if (menuToggle && primaryNav) {
      function setMenuOpen(open) {
        menuToggle.setAttribute("aria-expanded", String(open));
        primaryNav.classList.toggle("is-open", open);
      }

      setMenuOpen(false);
      menuToggle.addEventListener("click", () => {
        setMenuOpen(menuToggle.getAttribute("aria-expanded") !== "true");
      });
      primaryNav.querySelectorAll("a").forEach((link) => {
        link.addEventListener("click", () => setMenuOpen(false));
      });
      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && menuToggle.getAttribute("aria-expanded") === "true") {
          setMenuOpen(false);
          menuToggle.focus();
        }
      });
      document.addEventListener("click", (event) => {
        if (!primaryNav.contains(event.target) && !menuToggle.contains(event.target)) {
          setMenuOpen(false);
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
