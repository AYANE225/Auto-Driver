// Load GIFs only on request. Keep a single excerpt playing and offer a stop control.
(() => {
  const cards = [...document.querySelectorAll("[data-driving-gif]")];
  function stop(card) {
    const button = card.querySelector("button");
    const image = card.querySelector("img");
    image.src = card.dataset.poster;
    button.setAttribute("aria-pressed", "false");
    button.textContent = "播放 GIF";
  }
  for (const card of cards) {
    const button = card.querySelector("button");
    const image = card.querySelector("img");
    button.hidden = false;
    button.addEventListener("click", () => {
      const playing = button.getAttribute("aria-pressed") === "true";
      cards.forEach(stop);
      if (!playing) {
        image.src = card.dataset.drivingGif;
        button.setAttribute("aria-pressed", "true");
        button.textContent = "停止动画";
      }
    });
    image.addEventListener("error", () => {
      if (button.getAttribute("aria-pressed") === "true") {
        stop(card);
        button.textContent = "加载失败 · 重试";
      }
    });
  }
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) stop(entry.target);
      }
    });
    cards.forEach((card) => observer.observe(card));
  }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) cards.forEach(stop);
  });
})();
