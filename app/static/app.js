(function () {
  var statusEl = document.getElementById("backend-status");
  var collectedEl = document.getElementById("collected-status");
  var reportEl = document.getElementById("weekly-report");
  var reportPanel = document.getElementById("report-panel");
  var comparisonEl = document.getElementById("comparison-view");
  var comparePanel = document.getElementById("compare-panel");

  if (statusEl) {
    fetch("/api/health")
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then(function (data) {
        var keyState = data.openai_api_key_configured
          ? "configured"
          : "not configured";
        statusEl.dataset.state = "ok";
        statusEl.textContent =
          "Backend reachable. OpenAI API key: " + keyState + ".";
      })
      .catch(function () {
        statusEl.dataset.state = "error";
        statusEl.textContent =
          "Backend unreachable. This page needs the local Community Voices app.";
      });
  }

  if (collectedEl) {
    fetch("/api/corpus")
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then(function (data) {
        if (data.chunk_count > 0) {
          collectedEl.textContent =
            data.chunk_count +
            " comment chunks are stored locally. Re-run python -m app.ingest to refresh the sample.";
        } else {
          collectedEl.textContent =
            "No discussions have been collected yet. Run python -m app.ingest.";
        }
      })
      .catch(function () {
        collectedEl.textContent =
          "Could not read the local vector store. Start the app and try again.";
      });
  }

  if (reportEl) {
    fetch("/api/report")
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then(function (data) {
        if (data.status !== "ready" || !data.report) {
          reportEl.innerHTML = "<p>No report has been generated yet.</p>";
          return;
        }
        if (reportPanel) {
          reportPanel.classList.remove("empty");
        }
        reportEl.innerHTML = renderReport(data.report);
      })
      .catch(function () {
        reportEl.innerHTML =
          "<p>Could not load the weekly report. Start the app and try again.</p>";
      });
  }

  if (comparisonEl) {
    fetch("/api/comparison")
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then(function (data) {
        if (data.status !== "ready" || !data.comparison) {
          comparisonEl.innerHTML = "<p>No comparison has been generated yet.</p>";
          return;
        }
        if (comparePanel) {
          comparePanel.classList.remove("empty");
        }
        comparisonEl.innerHTML = renderComparison(data.comparison);
      })
      .catch(function () {
        comparisonEl.innerHTML =
          "<p>Could not load the comparison. Start the app and try again.</p>";
      });
  }

  function renderReport(report) {
    var sourcesById = {};
    (report.sources || []).forEach(function (source) {
      sourcesById[source.citation_id] = source;
    });
    var html = [];
    html.push('<p class="report-kicker">RAG-powered report</p>');
    html.push(
      "<p><strong>Report window:</strong> " +
        escapeHtml(formatDisplayUtc(report.window_start)) +
        " → " +
        escapeHtml(formatDisplayUtc(report.window_end)) +
        "</p>"
    );
    html.push(
      "<p><strong>Generated:</strong> " +
        escapeHtml(formatDisplayUtc(report.generated_at)) +
        "</p>"
    );
    html.push(
      "<p><strong>Bounded sample:</strong> " +
        Number(report.chunk_count || 0) +
        " chunks, " +
        Number(report.thread_count || 0) +
        " threads, " +
        Number(report.comment_count || 0) +
        " comments.</p>"
    );
    html.push("<p>" + escapeHtml(formatMetrics(report, true)) + "</p>");
    html.push('<p class="disclaimer">' + escapeHtml(report.limitation || "") + "</p>");

    html.push("<h3>What Hacker News discussed this week</h3>");
    (report.retrospective || []).forEach(function (section) {
      html.push("<article class=\"theme\">");
      html.push("<h4>" + escapeHtml(section.title) + "</h4>");
      html.push("<p>" + linkCitations(section.body, sourcesById) + "</p>");
      html.push("</article>");
    });

    html.push("<h3>What Hacker News may discuss next week</h3>");
    html.push('<p class="hint">The following items are predictions, not observations.</p>');
    html.push("<ul class=\"predictions\">");
    (report.predictions || []).forEach(function (item) {
      html.push("<li>");
      html.push(
        "<p><strong>Prediction:</strong> " + linkCitations(item.claim, sourcesById) + "</p>"
      );
      if (item.rationale) {
        html.push("<p>" + linkCitations(item.rationale, sourcesById) + "</p>");
      }
      if (item.uncertainty) {
        html.push("<p class=\"hint\">" + escapeHtml(item.uncertainty) + "</p>");
      }
      html.push("</li>");
    });
    html.push("</ul>");

    html.push("<h3>Sources</h3>");
    html.push("<ul class=\"sources\">");
    (report.sources || []).forEach(function (source) {
      var label = source.story_title || source.citation_id;
      var href = source.source_url || "";
      html.push("<li>");
      html.push("<span>[" + escapeHtml(source.citation_id) + "] </span>");
      if (href) {
        html.push(
          '<a href="' +
            escapeHtml(href) +
            '">' +
            escapeHtml(label) +
            "</a>"
        );
      } else {
        html.push(escapeHtml(label));
      }
      html.push(
        " · comment " +
          escapeHtml(String(source.comment_id)) +
          " · " +
          escapeHtml(source.created_at || "")
      );
      html.push("</li>");
    });
    html.push("</ul>");
    return html.join("");
  }

  function renderComparison(comparison) {
    var html = [];
    html.push('<p class="report-kicker">RAG-powered report vs No-RAG baseline</p>');
    html.push(
      "<p><strong>Report window:</strong> " +
        escapeHtml(formatDisplayUtc(comparison.window_start)) +
        " → " +
        escapeHtml(formatDisplayUtc(comparison.window_end)) +
        "</p>"
    );
    html.push(
      "<p><strong>Generation model:</strong> " +
        escapeHtml(comparison.model) +
        "</p>"
    );
    html.push('<div class="compare-grid">');
    html.push(renderArm(comparison.rag, true));
    html.push(renderArm(comparison.baseline, false));
    html.push("</div>");
    html.push("<h3>Observed differences (this run)</h3>");
    html.push("<ul class=\"observations\">");
    (comparison.observations || []).forEach(function (row) {
      html.push("<li>");
      html.push("<p><strong>" + escapeHtml(row.dimension) + "</strong></p>");
      html.push("<p>RAG: " + escapeHtml(row.rag) + "</p>");
      html.push("<p>Baseline: " + escapeHtml(row.baseline) + "</p>");
      html.push("</li>");
    });
    html.push("</ul>");
    if (comparison.manual_note) {
      html.push("<h3>Manual note</h3>");
      html.push("<p>" + escapeHtml(comparison.manual_note) + "</p>");
    }
    html.push('<p class="disclaimer">' + escapeHtml(comparison.limitations || "") + "</p>");
    return html.join("");
  }

  function renderArm(arm, isRag) {
    arm = arm || {};
    var sourcesById = {};
    (arm.sources || []).forEach(function (source) {
      sourcesById[source.citation_id] = source;
    });
    var html = [];
    html.push('<article class="compare-arm">');
    html.push('<p class="report-kicker">' + escapeHtml(arm.label || (isRag ? "RAG-powered report" : "No-RAG baseline")) + "</p>");
    html.push(
      "<p><strong>External evidence supplied:</strong> " +
        (arm.evidence_supplied ? "yes" : "no") +
        "</p>"
    );
    html.push(
      "<p><strong>Generated:</strong> " +
        escapeHtml(formatDisplayUtc(arm.generated_at)) +
        "</p>"
    );
    html.push("<p>" + escapeHtml(formatMetrics(arm.metrics || arm, isRag)) + "</p>");
    html.push('<p class="disclaimer">' + escapeHtml(arm.limitation || "") + "</p>");
    html.push("<h4>What Hacker News discussed this week</h4>");
    (arm.retrospective || []).forEach(function (section) {
      html.push("<p><strong>" + escapeHtml(section.title) + "</strong></p>");
      if (isRag) {
        html.push("<p>" + linkCitations(section.body, sourcesById) + "</p>");
      } else {
        html.push("<p>" + escapeHtml(section.body || "") + "</p>");
      }
    });
    html.push("<h4>What Hacker News may discuss next week</h4>");
    html.push('<p class="hint">The following items are predictions, not observations.</p>');
    html.push("<ul class=\"predictions\">");
    (arm.predictions || []).forEach(function (item) {
      html.push("<li>");
      if (isRag) {
        html.push(
          "<p><strong>Prediction:</strong> " + linkCitations(item.claim, sourcesById) + "</p>"
        );
        if (item.rationale) {
          html.push("<p>" + linkCitations(item.rationale, sourcesById) + "</p>");
        }
      } else {
        html.push(
          "<p><strong>Prediction:</strong> " + escapeHtml(item.claim || "") + "</p>"
        );
        if (item.rationale) {
          html.push("<p>" + escapeHtml(item.rationale) + "</p>");
        }
      }
      if (item.uncertainty) {
        html.push("<p class=\"hint\">" + escapeHtml(item.uncertainty) + "</p>");
      }
      html.push("</li>");
    });
    html.push("</ul>");
    if (isRag && arm.sources && arm.sources.length) {
      html.push("<h4>Sources</h4>");
      html.push("<ul class=\"sources\">");
      arm.sources.forEach(function (source) {
        var label = source.story_title || source.citation_id;
        html.push("<li>");
        html.push("<span>[" + escapeHtml(source.citation_id) + "] </span>");
        if (source.source_url) {
          html.push(
            '<a href="' +
              escapeHtml(source.source_url) +
              '">' +
              escapeHtml(label) +
              "</a>"
          );
        } else {
          html.push(escapeHtml(label));
        }
        html.push("</li>");
      });
      html.push("</ul>");
    }
    html.push("</article>");
    return html.join("");
  }

  function formatDisplayUtc(value) {
    if (!value) {
      return "";
    }
    var date = new Date(value);
    if (isNaN(date.getTime())) {
      return String(value);
    }
    var months = [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
    ];
    var hours = date.getUTCHours();
    var minutes = date.getUTCMinutes();
    return (
      months[date.getUTCMonth()] +
      " " +
      date.getUTCDate() +
      ", " +
      date.getUTCFullYear() +
      " · " +
      (hours < 10 ? "0" : "") +
      hours +
      ":" +
      (minutes < 10 ? "0" : "") +
      minutes +
      " UTC"
    );
  }

  function formatMetrics(metrics, isRag) {
    metrics = metrics || {};
    var latencyLabel = isRag
      ? "End-to-end report latency"
      : "Baseline generation latency";
    var latency =
      metrics.latency_seconds == null ? "unavailable" : metrics.latency_seconds + "s";
    var tokens =
      metrics.total_tokens == null ? "unavailable" : String(metrics.total_tokens);
    return (
      latencyLabel +
      ": " +
      latency +
      ". Writing-model tokens only; embedding usage is not included. Total: " +
      tokens +
      "."
    );
  }

  function linkCitations(text, sourcesById) {
    return escapeHtml(text || "").replace(
      /\[(S\d+(?:\s*,\s*S\d+)*)\]/g,
      function (_match, inner) {
        return inner.split(/,\s*/).map(function (id) {
          var source = sourcesById[id];
          if (!source || !source.source_url) {
            return "[" + escapeHtml(id) + "]";
          }
          return (
            '<a href="' +
            escapeHtml(source.source_url) +
            '">[' +
            escapeHtml(id) +
            "]</a>"
          );
        }).join(" ");
      }
    );
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
})();
