/* Keep live monitor rows stable while app.js refreshes their values. */
(() => {
  const descriptor = Object.getOwnPropertyDescriptor(Element.prototype, "innerHTML");
  if (!descriptor?.set || !descriptor.get) return;

  Object.defineProperty(Element.prototype, "innerHTML", {
    configurable: descriptor.configurable,
    enumerable: descriptor.enumerable,
    get: descriptor.get,
    set(value) {
      if (this.__nexusLastHTML === value) return;
      this.__nexusLastHTML = value;
      if (this.id !== "backtestMarkets" || !this.children.length || !String(value).includes("market-progress-row")) {
        descriptor.set.call(this, value);
        return;
      }

      const incoming = document.createElement("div");
      descriptor.set.call(incoming, value);
      const current = new Map([...this.children].map((row) => [row.dataset.asset, row]));
      for (const next of [...incoming.children]) {
        const asset = next.dataset.asset || next.querySelector("b")?.textContent || "";
        let row = current.get(asset);
        if (!row) {
          row = next.cloneNode(true);
          row.dataset.asset = asset;
          this.appendChild(row);
        } else {
          row.children[0].textContent = next.children[0]?.textContent || "";
          row.children[1].textContent = next.children[1]?.textContent || "";
          row.querySelector("progress").value = next.querySelector("progress")?.value || 0;
          row.querySelector("small").textContent = next.querySelector("small")?.textContent || "";
          current.delete(asset);
        }
      }
      for (const stale of current.values()) stale.remove();
    },
  });
})();

function renderBacktestStable(status, markets) {
  if (!status) return;
  const active = ["fetching", "learning", "evaluating", "queued"].includes(status.status);
  const progress = status.progress == null ? "" : ` · ${(Number(status.progress) * 100).toFixed(0)}%`;
  const market = status.market_index && status.markets_total ? ` · market ${status.market_index}/${status.markets_total}` : "";
  const stage = status.stage ? ` · ${status.stage}` : "";
  const label = ["complete", "complete_with_errors"].includes(status.status)
    ? `${status.status === "complete_with_errors" ? "Complete with unavailable markets" : "Complete"} · ${status.markets_total || 1} market(s) · ${status.markets_failed || 0} failed · ${status.trades_scored || 0} scored · ${status.wins || 0} wins / ${status.losses || 0} losses · ${(Number(status.win_rate || 0) * 100).toFixed(1)}% win rate`
    : `${String(status.status || "idle").toUpperCase()}${market}${stage}${progress} · ${status.orders_placed ?? 0} broker orders · live`;
  const statusNode = document.getElementById("backtestStatus");
  if (statusNode.textContent !== label) statusNode.textContent = label;
  const stop = document.getElementById("stopBacktest");
  stop.disabled = !(active || ["error", "cancelled"].includes(status.status));
  stop.textContent = active ? "Stop" : status.status === "error" ? "Reset" : "Clear";
  document.getElementById("startBacktest").disabled = active;

  const root = document.getElementById("backtestMarkets");
  const incoming = new Map((markets || []).map((item) => [String(item.asset), item]));
  if (incoming.size && !root.children.length) root.textContent = "";
  const current = new Map([...root.children].map((row) => [row.dataset.asset, row]));
  for (const [asset, item] of incoming) {
    let row = current.get(asset);
    if (!row) {
      row = document.createElement("div");
      row.className = "market-progress-row";
      row.dataset.asset = asset;
      row.innerHTML = "<b></b><span></span><div class=progress-track><i></i></div><small></small>";
      root.appendChild(row);
    }
    row.children[0].textContent = asset;
    row.children[1].textContent = `${String(item.status || "queued").toUpperCase()}${item.stage ? ` · ${item.stage}` : ""}`;
    row.children[2].firstElementChild.style.width = `${Math.max(0, Math.min(100, Number(item.progress || 0) * 100))}%`;
    row.children[3].textContent = item.detail || `${Math.round(Number(item.progress || 0) * 100)}%`;
    current.delete(asset);
  }
  for (const row of current.values()) row.remove();
  if (!incoming.size && root.textContent !== "No active market training.") root.textContent = "No active market training.";

  const result = document.getElementById("backtestResult");
  if (status.status === "learning") {
    result.textContent = `Training ${status.asset || ""} · ${status.detail || "Working"} · Learning candles: ${status.learning_candles || "—"} · Evaluation candles queued: ${status.evaluation_candles || "—"} · broker orders: 0`;
  } else if (status.status === "complete" || status.status === "complete_with_errors") {
    const lHours = status.learning_hours || 5;
    const eHours = status.evaluation_hours || 1;
    const scored = status.trades_scored || 0;
    const decisions = status.decisions || 0;
    const netProfit = Number(status.net_profit || 0).toFixed(2);
    const minC = Math.round(Number(status.min_confidence || 0.7) * 100);
    const maxC = status.max_confidence ? Math.round(Number(status.max_confidence) * 100) : null;
    const confNote = maxC ? ` (highest signal was ${maxC}%, below ${minC}% threshold · set threshold to 60% to execute trades)` : ` (all signals remained below ${minC}% threshold)`;
    if (status.markets) {
      if (scored === 0) {
        result.textContent = `Local replay complete · ${lHours}h learned + ${eHours}h evaluated across ${status.markets_total || status.markets.length} markets (${decisions} candidate bars analyzed) · 0 trades taken${confNote} · Net P&L: $${netProfit}`;
      } else {
        result.textContent = `Local replay complete · ${lHours}h learned + ${eHours}h evaluated across ${status.markets_total || status.markets.length} markets · ${scored} scored trades · ${(Number(status.win_rate || 0) * 100).toFixed(1)}% win rate · Net P&L: $${netProfit} · Orders placed: 0`;
      }
    } else {
      const buckets = (status.confidence_buckets || [])
        .filter((bucket) => bucket.trades)
        .map((bucket) => `${bucket.range}: ${(Number(bucket.win_rate) * 100).toFixed(1)}% (${bucket.trades})`)
        .join(" · ");
      if (scored === 0) {
        result.textContent = `Local replay complete · ${lHours}h learned + ${eHours}h evaluated (${decisions} candidate bars analyzed) · 0 trades taken${confNote} · Net P&L: $${netProfit}`;
      } else {
        result.textContent = `Local replay only · ${lHours}h learned + ${eHours}h evaluated · ${scored} scored · ${(Number(status.win_rate || 0) * 100).toFixed(1)}% win rate · payout ${Number(status.payout_percent || 0).toFixed(1)}% · break-even ${(Number(status.break_even_win_rate || 0) * 100).toFixed(1)}% · net P&L $${netProfit} · orders placed: 0${buckets ? ` · confidence bands — ${buckets}` : ""}`;
      }
    }
  } else if (status.status === "error") {
    result.textContent = `Backtest error: ${status.error || "unknown error"}`;
  }
}
