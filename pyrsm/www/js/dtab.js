/*
 * pyrsm.dtab: an interactive table in a notebook or a rendered (HTML) report.
 *
 * The same table Radiant's result views show (ResultTable.svelte, with the
 * helpers of resultTable.ts): Tabulator, a filter box above every column
 * (numbers take the Data > View shorthand 5..10, ..5, >5; text matches
 * anywhere; a column with a few distinct values gets a pick list), sorting that
 * starts descending, 10 rows a page (5 / 10 / 25 / 50 / All) and no pager for a
 * short table, label columns shaded, a totals row kept at the bottom outside
 * sorting and filtering, a bold Total column, Pivot's color bars or heat map,
 * a row count, "Clear filters" and "Download CSV".
 *
 * Radiant's tests compare these helpers with resultTable.ts.
 */
(function () {
  "use strict";

  function formatCell(value, kind, dec) {
    if (value === null || value === undefined || value === "") return "";
    if (typeof value !== "number") return String(value);
    if (!Number.isFinite(value)) return Number.isNaN(value) ? "NaN" : value > 0 ? "Inf" : "-Inf";
    if (kind === "percent") return (value * 100).toFixed(dec) + "%";
    if (Number.isInteger(value)) return String(value);
    var abs = Math.abs(value);
    if (abs > 0 && abs < Math.pow(10, -dec)) return value.toPrecision(3);
    return value.toFixed(dec);
  }

  function inferKind(values) {
    var present = values.filter(function (v) { return v !== null && v !== undefined && v !== ""; });
    return present.length > 0 && present.every(function (v) { return typeof v === "number"; }) ? "number" : "text";
  }

  var RANGE_RE = /^\s*(-?[\d.eE+]*?)\s*\.\.\s*(-?[\d.eE+]*)\s*$/;
  var CMP_RE = /^\s*(>=|<=|==|!=|>|<|=)\s*(-?[\d.eE+-]+)\s*$/;

  function numberMatches(filter, value) {
    var text = String(filter === null || filter === undefined ? "" : filter).trim();
    if (!text) return true;
    if (typeof value !== "number") return false;
    var range = RANGE_RE.exec(text);
    if (range) {
      var lo = range[1] === "" ? -Infinity : Number(range[1]);
      var hi = range[2] === "" ? Infinity : Number(range[2]);
      return value >= lo && value <= hi;
    }
    var cmp = CMP_RE.exec(text);
    if (cmp) {
      var x = Number(cmp[2]);
      switch (cmp[1]) {
        case ">": return value > x;
        case ">=": return value >= x;
        case "<": return value < x;
        case "<=": return value <= x;
        case "!=": return value !== x;
        default: return value === x;
      }
    }
    var y = Number(text);
    return Number.isFinite(y) ? value === y : false;
  }

  function textMatches(filter, value) {
    var text = String(filter === null || filter === undefined ? "" : filter).trim().toLowerCase();
    return !text || String(value === null || value === undefined ? "" : value).toLowerCase().indexOf(text) !== -1;
  }

  function heatBreaks(values) {
    var sorted = values.filter(function (v) { return Number.isFinite(v); }).sort(function (a, b) { return a - b; });
    if (!sorted.length) return [];
    var out = [];
    for (var k = 1; k <= 19; k++) {
      var h = (sorted.length - 1) * (k * 0.05);
      var lo = Math.floor(h);
      var next = sorted[Math.min(lo + 1, sorted.length - 1)];
      var q = sorted[lo] + (h - lo) * ((next === undefined ? sorted[lo] : next) - sorted[lo]);
      out.push(Math.round(q * 1e5) / 1e5);
    }
    return out;
  }

  function heatStyle(value, breaks) {
    if (!breaks.length) return {};
    var i = 0;
    while (i < breaks.length && value > breaks[i]) i++;
    var steps = breaks.length + 1;
    var shade = Math.round(255 - ((255 - 40) * i) / (steps - 1));
    return { backgroundColor: "rgb(255," + shade + "," + shade + ")" };
  }

  function barStyle(value, lo, hi) {
    var width = hi > lo ? Math.max(0, Math.min(100, ((value - lo) / (hi - lo)) * 100)) : 100;
    return {
      backgroundImage: "linear-gradient(90deg, lightblue " + width + "%, transparent " + width + "%)",
      backgroundSize: "98% 88%",
      backgroundRepeat: "no-repeat",
      backgroundPosition: "center",
    };
  }

  function countText(kept, all, total) {
    var unit = all === 1 ? "row" : "rows";
    var head = kept < all ? kept.toLocaleString() + " of " + all.toLocaleString() + " " + unit : all.toLocaleString() + " " + unit;
    return total && total > all ? head + " (the first " + all.toLocaleString() + " of " + total.toLocaleString() + ")" : head;
  }

  function build(root, spec) {
    var rows = spec.rows || [];
    var columns = spec.columns || [];
    var decimals = spec.decimals === undefined ? 3 : spec.decimals;
    var totals = spec.totals || null;
    var labelColumns = spec.labelColumns || [];
    var boldColumns = spec.boldColumns || [];
    var conditional = spec.conditional || "none";
    var conditionalColumns = spec.conditionalColumns || [];
    var pageSize = spec.pageSize || 10;
    var span = spec.span || null;

    function kindOf(col) {
      return col.kind || inferKind(rows.map(function (r) { return r[col.name]; }));
    }
    var values = [];
    rows.forEach(function (row) {
      conditionalColumns.forEach(function (c) {
        var v = row[c];
        if (typeof v === "number" && Number.isFinite(v)) values.push(v);
      });
    });
    var lo = values.length ? Math.min.apply(null, values) : 0;
    var hi = values.length ? Math.max.apply(null, values) : 0;
    var breaks = conditional === "heat" ? heatBreaks(values) : [];
    function paint(value, el) {
      if (typeof value !== "number" || !Number.isFinite(value)) return;
      if (conditional === "color_bar") Object.assign(el.style, barStyle(value, lo, hi));
      else if (conditional === "heat") Object.assign(el.style, heatStyle(value, breaks));
    }

    function columnDef(col) {
      var kind = kindOf(col);
      var isLabel = labelColumns.indexOf(col.name) !== -1;
      var isBold = boldColumns.indexOf(col.name) !== -1;
      var painted = conditional !== "none" && conditionalColumns.indexOf(col.name) !== -1;
      var base = {
        field: col.name,
        title: col.name,
        headerSort: true,
        headerSortStartingDir: "desc",
        resizable: true,
        minWidth: 80,
        cssClass: [isLabel ? "rsm-rt-label" : "", isBold ? "rsm-rt-bold" : ""].filter(Boolean).join(" "),
      };
      if (totals) {
        base.bottomCalc = function () { var v = totals[col.name]; return v === undefined ? "" : v; };
        base.bottomCalcFormatter = function (cell) { return formatCell(cell.getValue(), kind, decimals); };
      }
      if (kind === "number" || kind === "percent") {
        base.hozAlign = "right";
        base.headerHozAlign = "right";
        base.sorter = "number";
        base.formatter = function (cell) {
          var value = cell.getValue();
          if (painted) paint(value, cell.getElement());
          return formatCell(value, kind, decimals);
        };
        base.headerFilter = "input";
        base.headerFilterPlaceholder = kind === "percent" ? ">5 (%)" : "5..10, >5";
        base.headerFilterFunc = function (filter, value) {
          return numberMatches(filter, kind === "percent" && typeof value === "number" ? value * 100 : value);
        };
        return base;
      }
      var seen = {};
      var levels = [];
      rows.forEach(function (r) {
        var v = r[col.name];
        if (v === null || v === undefined) return;
        var s = String(v);
        if (!seen[s]) { seen[s] = true; levels.push(s); }
      });
      if (levels.length > 1 && levels.length <= 25 && levels.length < rows.length) {
        base.headerFilter = "list";
        base.headerFilterParams = { values: levels, multiselect: true, clearable: true };
        base.headerFilterFunc = function (filter, value) {
          return !Array.isArray(filter) || filter.length === 0 || filter.indexOf(String(value)) !== -1;
        };
        base.headerFilterEmptyCheck = function (value) {
          return value === undefined || value === null || value === "" || (Array.isArray(value) && value.length === 0);
        };
        return base;
      }
      base.headerFilter = "input";
      base.headerFilterFunc = function (filter, value) { return textMatches(filter, value); };
      return base;
    }

    var defs = columns.map(columnDef);
    var layout = defs;
    if (span && span.columns && span.columns.length) {
      var inSpan = {};
      span.columns.forEach(function (c) { inSpan[c] = true; });
      var first = defs.findIndex(function (d) { return inSpan[String(d.field)]; });
      layout = defs.slice(0, first).filter(function (d) { return !inSpan[String(d.field)]; })
        .concat([{ title: span.title, headerHozAlign: "center", columns: defs.filter(function (d) { return inSpan[String(d.field)]; }) }])
        .concat(defs.slice(first).filter(function (d) { return !inSpan[String(d.field)]; }));
    }

    var mount = root.querySelector(".rsm-result-table__mount");
    var count = root.querySelector(".rsm-dtab-count");
    var clear = root.querySelector(".rsm-dtab-clear");
    var download = root.querySelector(".rsm-dtab-download");
    mount.innerHTML = "";
    var paged = rows.length > pageSize;
    var table = new window.Tabulator(mount, {
      data: rows.map(function (r) { return Object.assign({}, r); }),
      layout: "fitDataTable",
      placeholder: "no rows",
      pagination: paged,
      paginationSize: pageSize,
      paginationSizeSelector: [5, 10, 25, 50, true],
      paginationCounter: "rows",
      paginationButtonCount: 5,
      columnHeaderVertAlign: "bottom",
      nestedFieldSeparator: false,
      columns: layout,
    });
    function shown(kept) {
      count.textContent = countText(kept, rows.length, spec.totalRows);
      clear.style.display = kept < rows.length ? "" : "none";
    }
    shown(rows.length);
    table.on("dataFiltered", function (_filters, kept) { shown(kept.length); });
    clear.addEventListener("click", function () { table.clearHeaderFilter(); });
    download.addEventListener("click", function () {
      var name = String(spec.filename || "table").replace(/[^A-Za-z0-9_.-]+/g, "_") || "table";
      table.download("csv", name + ".csv", {}, "active");
    });
    root.classList.add("rsm-dtab--live");
  }

  window.rsmDtabHelpers = {
    formatCell: formatCell, inferKind: inferKind, numberMatches: numberMatches,
    textMatches: textMatches, heatBreaks: heatBreaks, heatStyle: heatStyle, barStyle: barStyle,
  };

  /* Build every table on the page that is waiting for Tabulator. */
  window.rsmDtabRun = function () {
    if (!window.Tabulator) return;
    var pending = document.querySelectorAll(".rsm-dtab:not(.rsm-dtab--live)");
    for (var i = 0; i < pending.length; i++) {
      var root = pending[i];
      var data = root.querySelector("script.rsm-dtab-spec");
      if (!data) continue;
      try {
        build(root, JSON.parse(data.textContent));
      } catch (err) {
        root.classList.add("rsm-dtab--live");
        if (window.console) console.error("pyrsm.dtab", err);
      }
    }
  };
})();
