"use strict";

/**
 * Returns an HTML table selectable uniprots, given a list of uniprot data.
 */
function makeProtTable(prots, wsid, opts) {
    opts = opts || {};
    let rows = "";
    for (const entry of prots) {
        const checkbox = `<td><input uniprot='${entry.uniprot}' type='checkbox' checked /></td>`;
        const copybtn = `<td><button type='button' class='btn btn-default btn-sm' onclick="navigator.clipboard.writeText('${entry.uniprot}')">Copy</button></td>`;
        rows += `
        <tr>
            ${opts.skipSelect ? '' : checkbox}
            <td><a href='/${wsid}/protein/${entry.uniprot}'>${entry.uniprot}</a></td>
            <td>${entry.gene}</td>
            <td>${entry.name}</td>
            ${opts.addCopyButton ? copybtn : ''}
        </tr>
            `;
    }
    let selectAll = `
        <a id='deselect-all' style='cursor:pointer'>deselect all</a> / <a id='select-all' style='cursor:pointer'>select all</a>
        `;
    let selectHeader = '<th></th>';
    if (opts.skipSelect) {
        selectAll = '';
        selectHeader = '';
    }
    let copyHeader = '';
    if (opts.addCopyButton) {
		copyHeader = '<th>Uniprot \u27A1 Clipboard</th>';
    }

    let html=`
    <div>
        ${selectAll}
        <table class='table table-bordered table-hover table-condensed'>
            <thead><tr class='active'>${selectHeader}<th>Uniprot</th><th>Gene</th><th>Name</th>${copyHeader}</tr></thead>
            <tbody>
                ${rows}
            </tbody>
        </table>
    </div>
    `;

    const setAll = (checked) => {
        jqTable.find('input').each((idx, input) => {
            input.checked = checked;
        });
    }

    const jqTable = $(html);
    if (!opts.skipSelect) {
        jqTable.find('#select-all')[0].addEventListener('click', () => setAll(true));
        jqTable.find('#deselect-all')[0].addEventListener('click', () => setAll(false));
    }
    return jqTable[0];
}

/**
 * Returns a list of selected uniprot ids within this element.
 */
function getSelectedProts(el) {
    const selected = [];
    $(el).find('input').each((idx, input) => {
        if (input.checked) {
            selected.push(input.getAttribute('uniprot'));
        }
    });
    return selected;
};

/**
 * Returns a button element that, when clicked, will get selected uniprots
 * and add them to the output text element.
 */
function makeAddButton(inputEl, outputEl) {
    // Create the button HTML.
    const el = $("<span>Click to add selected prots: <button>Pre-load</button></span>")

    // Setup what the button does.
    el[0].addEventListener('click', () => {
        // Grab the uniprot IDs that are currently selected.
        const uniprots = getSelectedProts(inputEl);

        // Add a newline if there's already something in the text box.
        const cur = outputEl.value;
        if (cur.length > 0 && cur[cur.length - 1] != '\n') {
            outputEl.value += '\n';
        }

        // Put our uniprots in there.
        outputEl.value += uniprots.join('\n');

        // Scroll to the bottom of the text box.
        outputEl.scrollTop = outputEl.scrollHeight;
    });

    return el[0];
}


/**
 * Configures a "prot search" widget.
 * input: The text element for entering search terms.
 * button: The search button.
 * resultsDisplayEl: Where to draw the search results.
 * outputEl: Where to write any selected uniprots to.
 * ws_id: Workspace id
 */
function setupProtSearch(input, button, resultsDisplayEl, outputEl, ws_id, opts) {
    opts = opts || {};
    resultsDisplayEl.style = "display: inline-block; margin: 1rem;"

    input.addEventListener('keyup', (e) => {
        // If someone hits "ENTER" in the search box, treat it as clicking
        // the search button.
        if (e.keyCode == 13) {
            button.click();
        }
    });


    button.addEventListener('click', () => {
        resultsDisplayEl.innerHTML = '<span class="loader"></span>';
        const query = encodeURIComponent(input.value);
        fetch(`/api/prot_search/?search=${query}`).then((resp) => {
            return resp.json()
        }).then((respJson) => {
            const table = makeProtTable(respJson.matches, ws_id, opts);
            const el = $('<div></div>');
            if (respJson.reached_limit) {
                el[0].appendChild($('<div class="alert alert-warning">Query matched too many prots, not all shown</div>')[0]);
            }
            if (!opts.skipAddButton) {
                const addButton = makeAddButton(table, outputEl);
                el[0].appendChild(addButton);
            }
            el[0].appendChild(table);
            resultsDisplayEl.innerHTML = "";
            resultsDisplayEl.appendChild(el[0]);
        });
    });
}


/**
 * Configures a Global Data parse & prot search widget.
 * input: The text element containing the pasted global data information.
 * button: The search button.
 * bulkResultsDisplay: Where to draw the search results.
 * outputEl: Where to write any selected uniprots to.
 */
function setupGlobalDataProtSearch(input, button, bulkResultsDisplay, outputEl, wsid) {
    /**
     * Renders the parsed protein lookup results (data) to outEl.
     */
    function showResultsTable(outEl, data, addButton) {
        const structure = data.parsed;
        const targetData = data.targetData;

        const drugsWithNoData = [];
        const targetsWithNoData = [];
        const targets = {};

        /**
         * Gathers data out of the structured targetQuery and the returned
         * targetData.
         */
        function setupTarget(targetQuery) {
            const out = {
                name: targetQuery.name,
                protdata: [],
                ids: targetQuery.ids, 
                drugs: [],
            }
            for (const targetId of targetQuery.ids) {
                if (targetId in targetData && targetData[targetId].length > 0) {
                    for (const [prot, name] of targetData[targetId]) {
                        out.protdata.push({
                            uniprot: prot,
                            gene: targetId,
                            name: name,
                        });
                    }
                }
            }
            if (out.protdata.length == 0) {
                targetsWithNoData.push(out);
            }
            return out;
        }

        // Go through the structured query to find all the drugs and targets.
        // We'll mark any drugs without any proteins found, and setup
        // our target data.
        for (const entry of structure) {
            let drugHasAnyData = false;
            for (const target of entry.targets) {
                if (!(target.name in targets)) {
                    targets[target.name] = setupTarget(target);
                }
                targets[target.name].drugs.push(entry.title);

                if (targets[target.name].protdata.length > 0) {
                    drugHasAnyData = true;
                }
            }
            if (!drugHasAnyData) {
                drugsWithNoData.push(entry);
            }
        }

        outEl.innerHTML = "";

        const summary = $(`
        <div class='panel panel-info'>
            <div class='panel-heading'><h4>Summary</h4></div>
        </div>
        `)[0];

        const summaryBody = $(`
        <div class='panel-body'>
            Targets: <b>${Object.keys(targets).length}</b><br>
            Drugs: <b>${structure.length}</b><br>
        </div>
        `)[0];

        let drugTable = `
        <table class='table table-condensed table-hover'
               style='width:30%; border: 1px solid #aaa; margin-top:0.5rem;
                      display: inline-table; margin-right: 2rem; vertical-align: top'>
            <thead><tr class='info'>
                <th colspan=2>Drugs with no proteins found: <b>${drugsWithNoData.length}</b></th></tr>
                <tr><th>Drug</th><th>Targets</th>
            </tr></thead>
            <tbody>
            `;
        for (const drug of drugsWithNoData) {
            const targetNames = [];
            for (const target of drug.targets) {
                targetNames.push(target.name);
            }
            drugTable += `<tr><td>${drug.title}</td><td>${targetNames.join(', ')}</td></tr>`;
        }
        drugTable += `</tbody></table>`;

        const emptyTargetTableRows = [];
        for (const target of targetsWithNoData) {
            const drugs = target.drugs.join(", ");
            const ids = target.ids.join(", ");
            const row = `<tr><td>${target.name}</td><td>${drugs}</td><td>${ids}</td></tr>`;
            emptyTargetTableRows.push(row);
        }

        let emptyTargetTable = `
            <table class='table table-condensed table-hover'
                style='width:60%; border: 1px solid #aaa; margin-top:0.5rem;
                       display: inline-table; vertical-align: top;'>
                <thead><tr class='info'>
                    <th colspan=3>Targets with no proteins found: <b>${targetsWithNoData.length}</b></th></tr>
                    <tr><th>Target</th><th>Drugs</th><th>IDs</th>
                </tr></thead>
                <tbody>
                    ${emptyTargetTableRows.join('')}
                </tbody>
            </table>
        `;
        summaryBody.appendChild($(drugTable)[0]);
        summaryBody.appendChild($(emptyTargetTable)[0]);
        summary.appendChild(summaryBody);
        outEl.appendChild(summary);

        // Now we setup the table displaying all of the targets.
        const targetTable = $("<table class='table table-condensed table-bordered'></table>")[0];
        const tableBody = $("<tbody></tbody>")[0];
        targetTable.appendChild(tableBody);

        // We want a particular ordering of targets in the table.
        // We first sort by number of drugs listing this particular target.
        // For ties, we sort by target name, which should hopefully group
        // similar targets together.
        let names = Object.keys(targets);
        names.sort((a, b) => {
            const ta = targets[a];
            const tb = targets[b];
            // Sort first by number of drugs.
            if (ta.drugs.length < tb.drugs.length) {
                return 1;
            } else if (ta.drugs.length > tb.drugs.length) {
                return -1;
            }
            // Then sort by target name.
            if (a < b) {
                return -1;
            } else if (a > b) {
                return 1;
            } else {
                return 0;
            }
        });
        
        // Go through all of our target data in order, constructing
        // the row for that target.
        for (const targetName of names) {
            const targetData = targets[targetName];
            const rowEl = $(`<tr style='max-width: 800px'></tr>`)[0];

            targetData.ids.sort();
            const ids = targetData.ids.map(
                id => `<span class='label label-info' style='background-color: #eef; color: black;'>${id}</span>`)
                .join(' ');
            targetData.drugs.sort();
            const drugs = targetData.drugs.map(
                drug => `<span class='label label-primary' style='background-color: #def; color: black;'>${drug}</span>`)
                .join(' ');

            const leftCell = $(`
                <td style='width:50%'>
                <div class=''>
                    <h4>${targetName}</h4>
                <h5>IDs: ${ids} </h5>
                <h5>Drugs: ${drugs} </h5>
                </div></td>`)[0];
            rowEl.appendChild(leftCell);
            const rightCell = $(`<td style='width:50%'></td>`)[0];
            rowEl.appendChild(rightCell);
            if (targetData.protdata.length > 0) {
                const table = makeProtTable(targetData.protdata, wsid);
                const tableWrap = $('<div style="display:inline-block"></div>')[0];
                tableWrap.appendChild(table);
                rightCell.appendChild(tableWrap);
            } else {
                rowEl.classList.add("warning");
            }
            tableBody.appendChild(rowEl);
        }

        const targetPanel = $(`
        <div class='panel panel-info'>
            <div class='panel-heading'><h4>Targets</h4></div>
        </div>`)[0];
        const targetPanelBody = $(`<div class='panel-body'></div>`)[0];
        targetPanelBody.appendChild(addButton);
        targetPanelBody.appendChild(targetTable);
        targetPanel.appendChild(targetPanelBody);
        outEl.appendChild(targetPanel);
    }

    button.addEventListener('click', () => {
        const query = input.value;

        bulkResultsDisplay.innerHTML = "";

        const addButton = makeAddButton(bulkResultsDisplay, outputEl);
        bulkResultsDisplay.appendChild($(`<hr/>`)[0]);

        const el = $("<div class='small'>Loading...</div>")[0];
        // We're only POST'ing because of data size limitations, we don't
        // really need CSRF protection, but it's not hard to just grab one
        // from the page.
        const csrf = document.getElementsByName("csrfmiddlewaretoken")[0].value
        const formData = new FormData()
        formData.append('csrfmiddlewaretoken', csrf)
        formData.append('query', query)
        const opts = {
            method: "POST",
            body: formData
        };
        fetch('/api/global_data_prot_search/', opts).then((resp) => {
            return resp.json()
        }).then((respJson) => {
            showResultsTable(el, respJson, addButton);
        });
        bulkResultsDisplay.appendChild(el);
        bulkResultsDisplay.appendChild(document.createElement("hr"));
    });
}
