class MultiInput {
    constructor(el, initialValues) {
        this.el = el;
        this._inputs = [];
        this._els = [];
        if (initialValues && initialValues.length > 0) {
            for (const initialValue of initialValues) {
                this.addInput(initialValue);
            }
        } else {
            // If we have no initial values, start with a blank.
            this.addInput();
        }
    }


    clear() {
        this._inputs = [];
        this._els = [];
        this.render();
    }

    getValues() {
        const out = [];
        for (const input of this._inputs) {
            const val = input.value.trim();
            if (val) {
                out.push(val);
            }
        }
        return out;
    }

    addInput(value) {
        const el = $(`
        <div class='input-group value-form' style='margin-right: 1rem'>
            <div id='remove'
                 class='input-group-addon glyphicon glyphicon-trash in-remove'/>
        </div>
            `)[0]
        const placeholder = "Value"
        const input = $(`<input class="form-control value-form" placeholder="${placeholder}"/>`)[0];
        el.prepend(input)
        const elel = $("<div  class='form-inline value-form'></div>")[0];
        elel.append(el)
        this._inputs.push(input);
        this._els.push(elel);
        if (value) {
            input.value = value;
        }

        el.querySelector('#remove').addEventListener('click', () => {
            const eidx = this._els.indexOf(elel);
            this._els.splice(eidx, 1);

            const iidx = this._inputs.indexOf(input);
            this._inputs.splice(iidx, 1);

            this.render()
        });

        this.render()
    }

    render() {
        this.el.innerHTML = "";
        for (const el of this._els) {
            this.el.appendChild(el)
        }
        const moreBtn = $("<button class='btn btn-sm btn-default'>Add More</button>")[0];
        moreBtn.addEventListener('click', () => this.addInput());
        this.el.appendChild(moreBtn);
    }

}

function updateKeepStyle(checkbox) {
    const checked = checkbox.checked;
    if (!checked) {
        checkbox.closest('tr').classList.add('list-removed');
    } else {
        checkbox.closest('tr').classList.remove('list-removed');
    }
}
function makeCheckboxListen(checkbox) {
    checkbox.addEventListener('change', () => {
        updateKeepStyle(checkbox);
    });
}

const NO_PROT_FOUND = '???'

function makeProtListen(protEl, geneEl) {
    protEl.addEventListener('change', async () => {
        geneEl.innerText = NO_PROT_FOUND;
        const resp = await fetch(`/api/uniprot/${protEl.value}/`)
        const json = await resp.json();
        geneEl.innerText = json.gene;
        geneEl.setAttribute('href', `/42/protein/${protEl.value}/`);
    });
}

function makeDirectionInput(rowId, initVal) {
    return `
        <span style='white-space:nowrap'><input type='radio' ${(initVal==1?'checked':'')} name='dir${rowId}' id='up${rowId}' value=1><label for='up${rowId}'>Up</label></span>
        <span style='white-space:nowrap'><input type='radio' ${(initVal==-1?'checked':'')} name='dir${rowId}' id='down${rowId}' value=-1><label for='down${rowId}'>Down</label></span>
        <span style='white-space:nowrap'><input type='radio' ${(initVal==0?'checked':'')} name='dir${rowId}' id='unknown${rowId}' value=0><label for='unknown${rowId}'>Unknown</label></span>
        `;
}

function makeEvidenceInput(rowId, initVal) {
    if (true) {
        return `<input id='evidence' class='form-control' type='number' value='${initVal}' step=0.1 max=1 min=0>`;
    }
    return `
        <input type='radio' name='ev${rowId}' id='strong${rowId}'><label for='strong${rowId}'>Strong</label>
        <input type='radio' name='ev${rowId}' id='weak${rowId}'><label for='weak${rowId}'>Weak</label>
        `;
}

function generateDpiJson(table) {
    const out = [];
    for (const tr of table.querySelectorAll('tr')) {
        const get = (id) => tr.querySelector(id);
        if (!get('#uniprot')) {
            // Probably header.
            continue;
        }
        const keep = get('#keep').checked;
        const isNew = tr.isNew;
        if (!keep && isNew) {
            // This is a newly added row that we decided not to keep, just omit.
            continue;
        }
        const direction = get('input[type=radio]:checked').value;
        const entry = {
            'uniprot': get('#uniprot').value.trim(),
            'gene': get('#gene').innerText,
            'c50': get('#c50').value.trim(),
            'ki': get('#ki').value.trim(),
            'evidence': get('#evidence').value.trim(),
            'source': get('#source').value.trim(),
            'keep': keep,
            'isNew': isNew,
            'isChanged': tr.isChanged,
            direction
        };

        for (const inp of tr.querySelectorAll('input')) {
            inp.classList.remove('field-error');
        }

        if (keep && (isNew || entry.isChanged)) {
            // Uniprot must have resolved.
            if (entry.uniprot == '' || entry.gene == NO_PROT_FOUND) {
                get('#uniprot').classList.add('field-error');
            }
            // Any non-blanks measurements must be numbers
            if (entry.c50 != '' && isNaN(entry.c50)) {
                get('#c50').classList.add('field-error');
            }
            if (entry.ki != '' && isNaN(entry.ki)) {
                get('#ki').classList.add('field-error');
            }
            if (entry.evidence != '' && (
                    isNaN(entry.evidence) ||
                    parseFloat(entry.evidence) < 0 ||
                    parseFloat(entry.evidence) > 1)) {
                get('#evidence').classList.add('field-error');
            }
            // At least one measurement must be present.
            if (entry.c50 == '' && entry.ki == '' && entry.evidence == '') {
                get('#evidence').classList.add('field-error');
                get('#ki').classList.add('field-error');
                get('#c50').classList.add('field-error');
            }

            // There must be a source if this is a change.
            if (!isValidSource(entry.source)) {
                get('#source').classList.add('field-error');
            }
        }
        for (const inp of tr.querySelectorAll('input')) {
            if (inp.classList.contains('field-error')) {
                throw "DPI validation error";
            }
        }

        out.push(entry);
    }

    return out;
}

function isValidSource(text) {
    // Allow anything that starts with one of our drug collections.
    const prefixes = [
        'SLK',
        'BDBM',
        'CHEMBL',
        'DB',
        'CAY',
    ];
    for (const prefix of prefixes) {
        if (text.startsWith(prefix)) {
            return true;
        }
    }
    // Make sure it looks vaguely URL'ish.
    // If it is a box link, make sure it's to the specific resource and
    // not just the folder.
    return text.trim().startsWith('http') &&
            text.trim() != "https://twoxar.app.box.com/folder/90056063354";
}

function generateAttrJson(table) {
    const out = [];
    for (const tr of table.querySelectorAll('tr')) {
        const get = (id) => tr.querySelector(id);
        if (!get('#source')) {
            // Probably header.
            continue;
        }
        const entry = tr.origData;
        entry['value'] = tr.getValue();
        entry['source'] = get('#source').value;
        entry['isChanged'] = tr.isChanged;

        get('#source').classList.remove('field-error');
        if (entry['value'] != '' && tr.isChanged) {
            if (!isValidSource(entry['source'])) {
                get('#source').classList.add('field-error');
                throw "Source not provided";
            }
        }
        out.push(entry);
    }

    return out;
}

function makePostOpts(data, extra) {
    const csrf = document.getElementsByName("csrfmiddlewaretoken")[0].value
    const formData = new FormData();
    formData.append('csrfmiddlewaretoken', csrf);
    if (data) {
        const queryStr = JSON.stringify(data);
        formData.append('query', queryStr);
    }
    if (extra) {
        for (const key in extra) {
            const val = extra[key];
            formData.append(key, val);
        }
    }
    const opts = {
        method: "POST",
        body: formData
    };
    return opts;
}

function setupEditPage(attrEl, attrData, dpiEl, dpiData, refDrugId, refProposalId, bestDumaKey) {
    const attrTable = setupAttrTable(attrEl, attrData, true);
    const dpiTable = setupDpiTable(dpiEl, dpiData, true);
    document.getElementById('propose').addEventListener('click', async () => {
        const dpiData = generateDpiJson(dpiTable);
        const attrData = generateAttrJson(attrTable);
        const query = {
            refDrugId: refDrugId,
            refProposalId: refProposalId,
            bestDumaKey: bestDumaKey,
            newDrugData: {
                dpi: dpiData,
                attrs: attrData,
                note: $('#note')[0].value,
            },
        };

        const opts = makePostOpts(query, {propose_btn: true});
        const resp = await fetch('/drugs/edit/', opts);
        window.location = '/drugs/review/';
    });
}

function setupAttrTable(el, data, forEdit) {
    const headers = [
        'Property Name',
        'Proposed Value',
        'Source (url)',
        'Other Values',
    ];

    el.innerHTML=`
    <table class='table table-condensed table-hover'>
        <thead>
            <th>${headers.join('</th><th>')}</th>
        </thead>
        <tbody>
        </tbody>
    </table>`;
    const table = el.querySelector('table');
    let rowId = 0
    function makeRow(rowData) {
        rowId += 1;
        const row = document.createElement('tr');
        row.origData = rowData;

        function get(name) {
            if (name in rowData) {
                return rowData[name];
            } else {
                return '';
            }
        }

        let els;
        const rawValue = get('value');
        let value;
        const other = get("other");
        const otherRows = []
        let prevValue = undefined;
        for (const [value, source] of other) {
            const srcSpan = `<span class='source source-${source}'>${source}</span>`
            if (value === prevValue) {
                otherRows[otherRows.length -1] += ' ' + srcSpan;
            } else {
                prevValue = value;
                otherRows.push(`${value} ${srcSpan}`);
            }
        }
        const otherText = otherRows.join('<hr>');
        if (forEdit) {
            if (Array.isArray(rawValue)) {
                value = '<span class="multi"></span>';
            } else {
                value = `<input id='value' value='${rawValue}' class='value-form form-control' type='text'>`;
            }
            els = [
            get("name"),
            value,
            `<textarea id='source' class='form-control' rows='1'>${get("source")}</textarea>`,
            `<div class='other-values' id='other'>${otherText}</div>`,
            ];

        } else {
            if (Array.isArray(rawValue)) {
                value = rawValue.join("<hr>");
            } else {
                value = rawValue;
            }
            els = [
                get("name"),
                value,
                get("source"),
                otherText,
            ];
        }
        for (const el of els) {
            const tdel = `<td>${el}</td>`;
            row.innerHTML += tdel
        }

        if (forEdit) {
            // Needs to occur after it has become real HTML.
            if (Array.isArray(rawValue)) {
                const otherValues = []
                for (const [value, src] of other) {
                    otherValues.push(value);
                }
                const multiInput = new MultiInput(
                    row.querySelector('.multi'),
                    otherValues,
                );
                row.getValue = () => {
                    return multiInput.getValues();
                };
            } else {
                row.getValue = () => {
                    return row.querySelector('#value').value;
                };
            }
        }

        if (rowData.isNew) {
            row.classList.add('list-new');
            row.isNew = true;
        }
        if (rowData.isChanged) {
            row.classList.add('list-changed');
        }

        table.querySelector('tbody').appendChild(row);

        function onInputChange(input) {
            let changed = false;
            for (const input of row.querySelectorAll('input')) {
                if (input.value != input.origValue) {
                    changed = true;
                    break;
                }
            }
            if (changed) {
                row.classList.add('list-changed');
                row.isChanged = true;
            } else {
                row.classList.remove('list-changed');
                row.isChanged = false;
            }
        }

        for (const input of row.querySelectorAll('input')) {
            input.origValue = input.value;
            input.addEventListener('change', () => {
                onInputChange(input);
            });
        }
    }

    for (const rowData of data) {
        makeRow(rowData);
    }

    return table;
}

function setupDpiTable(el, data, forEdit) {
    const headers = [
        'Keep',
        'Uniprot',
        'Gene',
        'Direction',
        'C50 (nM)',
        'Ki (nM)',
        'Evidence (0 to 1)',
        'Source / Reference (url)',
    ];

    el.innerHTML=`
    <table class='table table-condensed table-hover' style='width:auto'>
        <thead>
            <th>${headers.join('</th><th>')}</th>
        </thead>
        <tbody>
        </tbody>
    </table>`;
    const table = el.querySelector('table');
    let rowId = 0
    function makeRow(rowData) {
        rowId += 1;
        const row = document.createElement('tr');

        function get(name) {
            if (name in rowData) {
                return rowData[name];
            } else {
                return '';
            }
        }

        const checked = rowData.keep !== false ? 'checked' : '';


        let els;
        const uniprot = get('uniprot')
        // We aren't associated with any workspace here, so linking
        // to the protein is a bit tricky. Just pick an arbitrary one.
        const uniprotHref = `/42/protein/${uniprot}`;

        if (forEdit) {
            els = [
            `<input id='keep' type='checkbox' ${checked}>`,
            `<input id='uniprot' value='${get("uniprot")}' class='form-control' type='text'>`,
            `<a href='${uniprotHref}' target='_blank' id='gene'>${get("gene")}</a>`,
            makeDirectionInput(rowId, get("direction")),
            `<input id='c50' class='form-control ' type='text' value='${get("c50")}'>`,
            `<input id='ki' class='form-control' type='text' value='${get("ki")}'>`,
            makeEvidenceInput(rowId, get("evidence")),
            `<input id='source' class='form-control' style='width:300px' value='${get("source")}'>`,
            ];
        } else {
            els = [
                `<input id='keep' type='checkbox' ${checked} disabled>`,
                `<a href='${uniprotHref}' target='_blank'>${uniprot}</a>`,
                get("gene"),
                get("direction"),
                get("c50"),
                get("ki"),
                get("evidence"),
                get("source"),
            ];
        }
        for (const el of els) {
            const tdel = `<td>${el}</td>`;
            row.innerHTML += tdel
        }
        if (rowData.isNew) {
            row.classList.add('list-new');
            row.isNew = true;
        }
        if (rowData.isChanged) {
            row.classList.add('list-changed');
        }

        table.querySelector('tbody').appendChild(row);

        const keepCheckbox = row.querySelector('input[type=checkbox]');
        updateKeepStyle(keepCheckbox);
        if (forEdit) {
            makeCheckboxListen(keepCheckbox);
            makeProtListen(row.querySelector('#uniprot'), row.querySelector('#gene'));
        }



        function onInputChange(input) {
            let changed = false;
            for (const input of row.querySelectorAll('input')) {
                if (input.value != input.origValue || input.checked != input.origChecked) {
                    changed = true;
                    break;
                }
            }
            if (changed) {
                row.classList.add('list-changed');
                row.isChanged = true;
            } else {
                row.classList.remove('list-changed');
                row.isChanged = false;
            }
        }

        for (const input of row.querySelectorAll('input')) {
            input.origValue = input.value;
            input.origChecked = input.checked;
            input.addEventListener('change', () => {
                onInputChange(input);
            });
        }
    }

    for (const rowData of data) {
        makeRow(rowData);
    }

    if (forEdit) {
        const addEl = document.createElement('button');
        addEl.type = 'button';
        addEl.classList = "btn btn-default";
        addEl.innerText = 'Add DPI';

        addEl.addEventListener('click', () => {
            makeRow({isNew: true});
        });

        table.appendChild(addEl);
    }

    return table;
}

function insertAfter(el, ref) {
    ref.parentNode.insertBefore(el, ref.nextSibling);
}

function setupProposals(data, canModify) {
    function colorReviewRow(row) {
        const statusText = row.querySelector('td').innerText;
        for (const className of row.classList) {
            if (className.startsWith('status-')) {
                row.classList.remove(className);
            }
        }
        row.classList.add('status-' + statusText.replace(/ /g,'-'));
    }
    const curExpanded = {};
    async function toggleRow(row) {

        let expand = true;
        if (curExpanded.row == row) {
            expand = false;
        }

        if (curExpanded.row) {
            curExpanded.el.remove();
            curExpanded.row = null;
            curExpanded.el = null;
        }

        if (expand) {
            const resp = await fetch(`/drugs/proposal/${row.id}/`);
            const respObj = await resp.json();
            const reviewEl = $("<td class='review' colspan=100></td>")[0];
            const btnGroup = $("<div class='btn-group' />")[0];
            function onEdit() {
                window.open(`/drugs/edit/?prop_id=${row.id}`, '_blank');
            }
            async function changeState(reso) {
                const opts = makePostOpts();
                const resp = await fetch(`/drugs/resolve/${row.id}/${reso}/`, opts);
                if (resp.ok) {
                    const json = await resp.json()
                    newStatus = json.newStatus;
                } else {
                    newStatus = '!Error Updating!';
                    console.error("Failed to update status", resp);
                }
                row.querySelector('td').innerText = newStatus;
                colorReviewRow(row);
            }
            const resolveOpts = [
                [() => changeState(2), 'Accept', 'primary'],
                [() => changeState(1), 'Reject', 'danger'],
                [onEdit, 'Edit', 'info'],
                [() => changeState(3), 'Skip', 'default'],
            ];
            for (const resolveOpt of resolveOpts) {
                const [onClick, text, type] = resolveOpt;
                const disabled = (canModify || text == 'Edit') ? '' : 'disabled'
                const btn = $(`<button ${disabled} class='btn btn-${type}'>${text}</button>`)[0];
                btn.addEventListener('click', onClick);
                btnGroup.appendChild(btn);
            }
            let titleContent;
            if (respObj.refUrl) {
                // This assumes canonical comes first in attrs, which is always
                // true for now, at least.
                titleContent = `Replacing <a href='${respObj.refUrl}'>${respObj.newDrug.attrs[0].value}</a>`;
            } else {
                titleContent = `Creating ${respObj.newDrug.attrs[0].value}`;
            }
            const title = $(`<h3>${titleContent}</h3>`)[0]
            reviewEl.appendChild(title);


            reviewEl.appendChild(btnGroup);

            if (respObj.validation.error) {
                const validation = $(`<div class='alert alert-danger'>${respObj.validation.error}</div>`)[0]
                reviewEl.appendChild(validation);
            }

            if (respObj.validation.matches && respObj.validation.matches.length > 0) {
                const validation = $(`<div class='alert alert-info'>SMILES matches existing drugs</div>`)[0]
                for (const match of respObj.validation.matches) {
                    const matchEl = $(`<li>${match.tag__value} (${match.collection__name})</li>`)[0];
                    validation.appendChild(matchEl);
                }
                reviewEl.appendChild(validation);
            }


            const attrTable = $('<div class="review-section"></div>')[0];
            reviewEl.appendChild(attrTable);
            const leftEl = document.createElement('td');
            const rightEl = document.createElement('td');
            reviewEl.appendChild(leftEl);
            reviewEl.appendChild(rightEl);
            const noteText = respObj.newDrug.note || '';

            const noteEl = $(`<div class='review-section'>
                <h3>Note</h3>
                <hr/>
                <div style='white-space: pre-wrap'>${noteText}</div>
                `)[0];
            reviewEl.appendChild(noteEl);

            leftEl.innerHTML = `
            <div class='review-section'>
                <h3>Proposed DPI</h3>
                <span id='main'></span>
            </div>
            `;
            rightEl.innerHTML = `
            <div class='review-section'>
                <h3>Existing DPI</h3>
                <span id='main'></span>
            </div>
            `;
            insertAfter(reviewEl, row);
            setupDpiTable(leftEl.querySelector('#main'), respObj.newDrug.dpi, false);
            setupDpiTable(rightEl.querySelector('#main'), respObj.refDrug, false);
            setupAttrTable(attrTable, respObj.newDrug.attrs, false);
            curExpanded.row = row;
            curExpanded.el = reviewEl;
        }
    }
    const el = document.getElementById('proposals');
    const tbody = el.querySelector('tbody')

    for (const rowData of data) {
        const cells = [rowData['state'], rowData['user'], rowData['date'], rowData['drug_name']];
        const id = rowData['id'];
        const data = `<tr id='${id}'><td>${cells.join('</td><td>')}</td></tr>`;
        tbody.innerHTML += data;
    }

    for (const row of tbody.querySelectorAll('tr')) {
        row.addEventListener('click', () => toggleRow(row));
        colorReviewRow(row);
    }
}

function preselectProposal(propId) {
    const tr = document.querySelector(`tr[id='${propId}'`);
    tr.click();
    tr.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
    });
}
