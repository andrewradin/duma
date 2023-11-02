
function insertAfter(el, ref) {
    ref.parentNode.insertBefore(el, ref.nextSibling);
}

function selectEnglish(localized_data) {
    console.info("Selecting from", localized_data);
    if (!localized_data || localized_data.length == 0) {
        return '';
    }
    for (const entry of localized_data) {
        if (entry.language == 'en') {
            return entry.text;
        }
    }
    return localized_data[0].text;
}

class PatentResolve {
    constructor(ws, searchId, resolveOpts, tableEl) {
        this.ws = ws;
        this.searchid = searchId;
        this.tableEl = tableEl;
        this.resolveOpts = resolveOpts;
        this._rowsById = {}

        for (const a of tableEl.querySelectorAll('a')) {
            a.addEventListener('click', (e) => e.stopPropagation());
        }

        for (const row of tableEl.querySelector("tbody").querySelectorAll("tr")) {
            const id = row.getAttribute('searchResultId');
            this._rowsById[id] = row;
            row.addEventListener('click', () => {
                this.displayPatent(id);
            });
            this._colorRow(row);
        }

        this.displayNextUnresolved(null);
    }

    _colorRow(row) {
        const statusEl = row.querySelectorAll('td')[1];
        const mapping = {
            'Irrelevant Drug': 'irrelevant',
            'Irrelevant Disease': 'irrelevant',
            'Irrelevant All': 'irrelevant',
            'Needs More Review': 'review',
            'Skipped': 'review',
        };
        let statusText = statusEl.textContent;
        if (statusText in mapping) {
            statusText = mapping[statusText];
        }
        statusEl.setAttribute('class', `status-${statusText}`);
    }

    displayNextUnresolved(fromRow) {
        const tableEl = this.tableEl;
        let reachedFrom = !fromRow;
        for (const row of tableEl.querySelector("tbody").querySelectorAll("tr")) {
            if (row == fromRow) {
                reachedFrom = true;
                continue;
            } else if (!reachedFrom) {
                continue;
            } else {
                if (row.querySelectorAll("td")[1].textContent == 'Unresolved') {
                    this.displayPatent(row.getAttribute('searchResultId'));
                    break;
                }
            }
        }
    }

    async displayPatent(id) {
        const rowEl = this._rowsById[id];
        rowEl.classList.add('selected-row')
        const details = await this._fetchPatentDetails(id);
        this._clearPatentDetails();

        if (this._prevId != id) {
            const patentDetailsEl = this._makePatentDetails(id, details);
            insertAfter(patentDetailsEl, rowEl);
            this._prevId = id;
        } else {
            this._prevId = undefined;
        }

        // Prefetch next row after completion?
    }

    async _fetchPatentDetails(id) {
        const results = await fetch(`/pats/${this.ws}/patent_details/${id}/`)
        if (results.ok) {
            return results.json();
        } else {
            console.info("Failure", results)
            throw "Failed to fetch patent";
        }
    } 

    _clearPatentDetails() {
        if (this._patentDetailsEl) {
            this._patentDetailsEl.remove();

        }
        if (this._prevId) {
            const row = this._rowsById[this._prevId];
            row.classList.remove('selected-row');
        }
    }

    _formatEv(evData) {
        if (!evData) {
            return ""
        }
        const tag = "<span class='evidence'>"
        return tag + evData.join("</span>"+tag) + "</span>";
    }

    _makePatentDetails(id, details) {
        const el = document.createElement('td');
        el.setAttribute('colspan', 10);
        if (details.available) {
            let claims = '';
            if (details.claims_localized.length > 0) {
                claims = selectEnglish(details.claims_localized);
            }
            claims = claims.replace(/\s*\n\s*/g, '<br/>');
            const drugEv = this._formatEv(details.evidence.drug_ev);
            const diseaseEv = this._formatEv(details.evidence.disease_ev);

            el.innerHTML = `
            <div class='row'>
            <div class='col-md-6'>
            <h3>${selectEnglish(details.title_localized)}</h3>
            <h4>Abstract</h4>
            <p class='patent-content'>${selectEnglish(details.abstract_localized)}</p>
            <h4>Claims</h4>
            <p class='patent-content'>${claims}</p>
            </div>
            <div class='col-md-6'>
            <h4>Mentions of drug</h4>
                ${drugEv}
            <br/>
            <h4>Mentions of disease</h4>
                ${diseaseEv}
            <br/>
            <h4>Metadata</h4>
            <ul>
            <li>Family: <b>${details.family_id}</b></li>
            <li>Assignee: <b>${details.assignee}</b></li>
            </ul>
            </div>
            </div>
            `;
        } else {
            el.innerHTML = `
            <div class='row'><div class='col-md-12'><h4>All data missing, refer to link above</h4></div></div>
            `;
        }
        el.classList.add('resolve-view');
        const btnGroup = $("<div class='btn-group' />")[0];
        for (const resolveOpt of this.resolveOpts) {
            const [enumId, text, type] = resolveOpt;
            const btn = $(`<button class='btn btn-${type}'>${text}</button>`)[0];
            btn.addEventListener('click', () => {
                this.resolvePatent(id, enumId);
            });
            btnGroup.appendChild(btn);
        }
            
        el.prepend(btnGroup);
        
        this._patentDetailsEl = el;
        return el;
    }

    async _resolveOnServer(id, resolution) {
        const row = this._rowsById[id];
        const statusEl = row.querySelectorAll('td')[1];
        statusEl.innerHTML="<span class='loader'></span>";

        const csrf = document.getElementsByName("csrfmiddlewaretoken")[0].value
        const formData = new FormData()
        formData.append('csrfmiddlewaretoken', csrf)
        const opts = {
            method: "POST",
            body: formData
        };

        const url = `/pats/${this.ws}/resolve/${id}/${resolution}/`;
        const resp = await fetch(url, opts);
        let newStatus = 'Unset';
        if (resp.ok) {
            const json = await resp.json()
            newStatus = json.newStatus;
        } else {
            newStatus = '!Error Updating!';
            console.error("Failed to update status", resp);
        }
        statusEl.textContent = newStatus;
        this._colorRow(row);
    }

    resolvePatent(id, resolution) {
        this._resolveOnServer(id, resolution);
        this._clearPatentDetails();
        this.displayNextUnresolved(this._rowsById[id]);
    }
}
