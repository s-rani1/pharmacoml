function setupCopyButtons() {
  document.querySelectorAll("pre").forEach((block) => {
    if (block.parentElement?.classList.contains("copy-wrap")) {
      return;
    }
    const wrapper = document.createElement("div");
    wrapper.className = "copy-wrap";
    block.parentNode.insertBefore(wrapper, block);
    wrapper.appendChild(block);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-button";
    button.textContent = "Copy";
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(block.innerText);
        button.textContent = "Copied";
        window.setTimeout(() => {
          button.textContent = "Copy";
        }, 1400);
      } catch (_) {
        button.textContent = "Failed";
        window.setTimeout(() => {
          button.textContent = "Copy";
        }, 1400);
      }
    });
    wrapper.appendChild(button);
  });
}

function setupTabs(root = document) {
  root.querySelectorAll("[data-tabs]").forEach((group) => {
    const buttons = Array.from(group.querySelectorAll("[data-tab-target]"));
    const panels = Array.from(group.querySelectorAll("[data-tab-panel]"));
    if (!buttons.length || !panels.length) {
      return;
    }

    const activate = (targetId) => {
      buttons.forEach((button) => {
        const isActive = button.dataset.tabTarget === targetId;
        button.classList.toggle("active", isActive);
        button.setAttribute("aria-selected", String(isActive));
      });
      panels.forEach((panel) => {
        panel.hidden = panel.dataset.tabPanel !== targetId;
      });
    };

    buttons.forEach((button) => {
      button.addEventListener("click", () => activate(button.dataset.tabTarget));
    });

    const activeButton = buttons.find((button) => button.classList.contains("active")) || buttons[0];
    activate(activeButton.dataset.tabTarget);
  });
}

function setupWorkflowExplorer() {
  const container = document.querySelector("[data-workflow]");
  if (!container) {
    return;
  }

  const buttons = Array.from(container.querySelectorAll("[data-workflow-step]"));
  const title = container.querySelector("[data-workflow-title]");
  const body = container.querySelector("[data-workflow-body]");
  const code = container.querySelector("[data-workflow-code]");

  const activate = (button) => {
    buttons.forEach((item) => item.classList.toggle("active", item === button));
    title.textContent = button.dataset.title || "";
    body.textContent = button.dataset.body || "";
    code.textContent = button.dataset.code || "";
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => activate(button));
  });

  if (buttons[0]) {
    activate(buttons[0]);
  }
}

function setupBenchmarkExplorer() {
  const container = document.querySelector("[data-benchmark-explorer]");
  if (!container) {
    return;
  }

  const buttons = Array.from(container.querySelectorAll("[data-benchmark-case]"));
  const title = container.querySelector("[data-case-title]");
  const summary = container.querySelector("[data-case-summary]");
  const metrics = container.querySelector("[data-case-metrics]");

  const activate = (button) => {
    buttons.forEach((item) => item.classList.toggle("active", item === button));
    title.textContent = button.dataset.caseTitle || "";
    summary.textContent = button.dataset.caseSummary || "";
    metrics.innerHTML = `
      <div class="metric-card"><span class="metric-label">Tier</span><strong>${button.dataset.caseTier || "-"}</strong></div>
      <div class="metric-card"><span class="metric-label">Truth</span><strong>${button.dataset.caseTruth || "-"}</strong></div>
      <div class="metric-card"><span class="metric-label">Why it matters</span><strong>${button.dataset.caseReason || "-"}</strong></div>
    `;
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => activate(button));
  });

  if (buttons[0]) {
    activate(buttons[0]);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setupCopyButtons();
  setupTabs();
  setupWorkflowExplorer();
  setupBenchmarkExplorer();
});
