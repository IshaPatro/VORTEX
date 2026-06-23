// VORTEX explainer site — light interactivity (no dependencies)

(function () {
  "use strict";

  // --- Mobile nav toggle ---
  const toggle = document.getElementById("navToggle");
  const links = document.getElementById("navLinks");
  if (toggle && links) {
    toggle.addEventListener("click", () => links.classList.toggle("open"));
    links.querySelectorAll("a").forEach((a) =>
      a.addEventListener("click", () => links.classList.remove("open"))
    );
  }

  // --- Scroll reveal for sections ---
  const revealEls = document.querySelectorAll(".section");
  revealEls.forEach((el) => el.classList.add("reveal"));

  if ("IntersectionObserver" in window) {
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            obs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    revealEls.forEach((el) => obs.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("in"));
  }

  // --- Active link highlighting on scroll (scroll-spy) ---
  const navAnchors = Array.from(document.querySelectorAll(".nav-links a"));
  const sectionMap = navAnchors
    .map((a) => {
      const id = a.getAttribute("href").slice(1);
      const section = document.getElementById(id);
      return section ? { a, section } : null;
    })
    .filter(Boolean);

  if ("IntersectionObserver" in window && sectionMap.length) {
    const spy = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const match = sectionMap.find((m) => m.section === entry.target);
            if (match) {
              navAnchors.forEach((a) => a.classList.remove("active"));
              match.a.classList.add("active");
            }
          }
        });
      },
      { rootMargin: "-45% 0px -50% 0px" }
    );
    sectionMap.forEach((m) => spy.observe(m.section));
  }
})();
