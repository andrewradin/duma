
function assert(condition, text) {
    if (!condition) {
        throw new Error(text);
    }
}

/**
 * User-adjustable sequence of input boxes.
 */
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
        <div class='input-group' style='margin-right: 1rem'>
            <div id='remove'
                 class='input-group-addon glyphicon glyphicon-trash in-remove'/>
        </div>
            `)[0]
        const placeholder = "Name or Synonym"
        const input = $(`<input class="form-control" placeholder="${placeholder}"/>`)[0];
        el.prepend(input)
        const elel = $("<div style='display:inline-block' class='form-inline'></div>")[0];
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

/**
 * Component representing the search elements for a single drug.
 * Displays the title, remove button, list of terms, search button, etc.
 */
class DrugNames {
    constructor(drugData, delegate) {
        this.drugData = drugData || {}
        const drugNames = this.drugData.drug_terms || [];
        const targetNames = this.drugData.target_terms || [];
        const wsa_id = this.drugData.wsa;
        let canonical = this.drugData.name || '';
        if (canonical && wsa_id) {
            const ws_id = delegate.ws;
            canonical = `<a href='/mol/${ws_id}/annotate/${wsa_id}/' target='_blank'>
                         ${canonical}
                <span class='glyphicon glyphicon-new-window'></span>
                </a>`;
        }
        this.delegate = delegate;
        this.el = $('<div class="drug-name-set panel panel-default"/>')[0];
        this.el.innerHTML = `
            <div class='panel-heading'>
            <h4 class='panel-title'><button id='remove'
                 class='btn btn-sm glyphicon glyphicon-trash'
                 style='display: inline;'></button>
            ${canonical}</h4>
            </div>
            <div class='panel-body'>
            <b>Drug Names</b> <a id='clear-drug-names' href='#'>Clear All</a>
            <div id='drug-names' class='drug-terms'></div>
            <b>Targets</b> <a id='clear-target-names' href='#'>Clear All</a>
            <div id='drug-targets' class='drug-terms'></div>
            <b>Notes</b>
            <div class='drug-note'>${this.drugData.study_note || ''}</div>
            <div class='drug-note'>${this.drugData.global_note || ''}</div>
            </div>
            `;

        this.el.querySelector('#clear-drug-names').addEventListener('click', (e) => {
            this.namesMultiInput.clear();
            e.preventDefault();
        });
        this.el.querySelector('#clear-target-names').addEventListener('click', (e) => {
            this.targetsMultiInput.clear();
            e.preventDefault();
        });

        this.el.querySelector('#remove').addEventListener('click', () => {
            this.delegate.remove(this);
        });
        const namesInputEl = this.el.querySelector('#drug-names');
        this.namesMultiInput = new MultiInput(namesInputEl, drugNames);
        const targetsInputEl = this.el.querySelector('#drug-targets');
        this.targetsMultiInput = new MultiInput(targetsInputEl, targetNames);

        const previewLink = $("<button class='btn btn-xs btn-primary'>Preview Patents</button>")[0];
        previewLink.addEventListener('click', () => this.preview());
        this.el.appendChild(previewLink);
    }

    getValues() {
        const a1 = this.namesMultiInput.getValues()
        const a2 = this.targetsMultiInput.getValues()
        return a1.concat(a2);
    }

    getRepr() {
        const drugTerms = this.namesMultiInput.getValues();
        const targetTerms = this.targetsMultiInput.getValues();
        const name = this.drugData.name || drugTerms[0] || targetTerms[0];
        // We use whatever we got back as drugData as the default repr,
        // and then tack on our edits wrt terms.
        return Object.assign({}, this.drugData, {
            drug_terms: drugTerms,
            target_terms: targetTerms,
            name: name,
        });
    }

    preview() {
        const names = this.getValues();
        this.delegate.previewPatents(names);
    }
}

class Drugs {
    constructor(el, initialDrugs, delegate) {
        this.el = el;
        this.drugNames = []
        this.delegate = delegate;

        if (initialDrugs) {
            for (const drug of initialDrugs) {
                this.addDrug(drug);
            }
        } else {
            this.addDrug();
        }
    }

    getDrugList() {
        const out = []
        for (const drug of this.drugNames) {
            const drugRepr = drug.getRepr();
            if (drugRepr) {
                out.push(drugRepr);
            }
        }
        return out;
    }

    addDrug(drugData) {
        const delegate = {
            remove: (child) => {
                this.drugNames.splice(this.drugNames.indexOf(child), 1);
                this.render()
            },
            previewPatents: (names) => this.delegate.previewPatents(names),
            ws: this.delegate.ws

        };
        this.drugNames.push(new DrugNames(drugData, delegate));
        this.render();
    }

    render() {
        this.el.innerHTML = ``;
        this.el.classList.add('form-inline');
        for (const drugNames of this.drugNames) {
            this.el.appendChild(drugNames.el);
        }
        const moreBtn = $("<button class='btn btn-sm btn-default'>Add Blank Drug</button>")[0];
        moreBtn.addEventListener('click', () => this.addDrug());
        this.el.appendChild(moreBtn);
        const drugsetBtn = $(`<span class='btn-group'>
            <button class='btn btn-sm btn-default dropdown-toggle' data-toggle='dropdown'>Add DrugSet <span class='caret' /></button>
            <ul class='dropdown-menu'>
            </ul>
            </span>`)[0];
        const optsEl = drugsetBtn.querySelector('ul');
        // Could pull down whole list of options, but most seem irrelevant.
        const opts = [['Selected', 'selected'], ['KTs', 'kts']];
        for (const opt of opts) {
            const [dsLabel, dsId] = opt;
            const optEl = $(`<li><a href="#" onClick="return false;">${dsLabel}</a></li>`)[0];
            optEl.addEventListener('click', () => {
                this.delegate._addDrugSet(dsId);
            });
            optsEl.appendChild(optEl);
        }
        this.el.appendChild(drugsetBtn);

        const drugSearchEl = $(`
        <span class='input-group' style='margin-left: 1rem; width: 500px'>
            </span>`)[0];
        const onSelected = (selected) => {
            this.delegate._addDrug(selected.wsa_id);
        };
        drawMolSearch(drugSearchEl, this.delegate.ws, onSelected, '');
        this.el.appendChild(drugSearchEl);

        const searchDrugBtn = drugSearchEl.querySelector('button');
        const searchDrugInput = drugSearchEl.querySelector('input');
        const searchResults = drugSearchEl.querySelector('#search-results');

    }
}

class PatentSearch {
    constructor(ws, searchEl, defaultSearchData, pastSearchQueries) {
        this._localStorageKey = 'patent-search-' + ws;
        this._setupLoadSettings(pastSearchQueries);

        // The defaultSearchData we are provided comes from the workspace,
        // mostly around disease names.
        // We override this with anything we have stored for this session.
        // We still want the original stored in case you want to reset.
        this._origDefaultSearchData = defaultSearchData;
        const data = localStorage.getItem(this._localStorageKey);
        if (data) {
            defaultSearchData = JSON.parse(data);
        }

        this.ws = ws;
        this.searchEl = searchEl;
        this._setupInputs(defaultSearchData);

        const searchBtn = searchEl.querySelector('#search-btn');
        searchBtn.addEventListener('click', () => this.search());
        this.searchBtn = searchBtn;

        // Saving state is local and fast, so we can be aggressive about it.
        document.addEventListener('keyup', () => this._saveState());
        document.addEventListener('click', () => this._saveState());
    }

    async _addDrugSet(drugsetName) {
        const resp = await fetch(`/pats/${this.ws}/drugset/${drugsetName}/`);
        if (!resp.ok) {
            const respText = await resp.text();
            showModal('Error', respText)
            return;
        }
        const respJson = await resp.json();
        for (const drug of respJson['drugs']) {
            this.drugs.addDrug(drug);
        }
        this._saveState();
    }

    async _addDrug(wsaId) {
        const resp = await fetch(`/pats/${this.ws}/drug/${wsaId}/`);
        if (!resp.ok) {
            const respText = await resp.text();
            showModal('Error', respText)
            return;
        }
        const drug = await resp.json();
        this.drugs.addDrug(drug);
        this._saveState();
    }

    async _fetchSuggestions(query) {
        console.info("Fetching suggestions for ", query);
        const resp = await fetch(`/pats/${this.ws}/search_drugs/${query}/`);
        if (!resp.ok) {
            const respText = await resp.text();
            showModal('Error', respText)
            return;
        }
        const respJson = await resp.json();
        return respJson['names'];
    }

    _setupLoadSettings(pastSearchQueries) {
        for (const loadBtn of document.querySelectorAll('#load-settings-btn')) {
            loadBtn.addEventListener('click', () => {
                const searchId = loadBtn.getAttribute('searchId');
                const data = pastSearchQueries[searchId];
                this._setupInputs(data);
            });
        }
    }

    _setupInputs(defaultSearchData) {
        const searchEl = this.searchEl;
        const diseaseNames = new MultiInput(
            searchEl.querySelector('#disease-names'),
            defaultSearchData.diseaseNames);
        diseaseNames.render();
        this.diseaseNames = diseaseNames;

        searchEl.querySelector('#reload-dis-names').addEventListener('click', () => {
            const diseaseNames = new MultiInput(
                searchEl.querySelector('#disease-names'),
                this._origDefaultSearchData.diseaseNames);
            diseaseNames.render();
            this.diseaseNames = diseaseNames;
        });

        const drugs = new Drugs(
            searchEl.querySelector('#drug-names'),
            defaultSearchData.drugList,
            this);
        drugs.render()
        this.drugs = drugs;
    }

    resultsPreviewHtml(data) {
        const itemHtmls = [];
        if (!data.items) {
            data.items = [];
        }
        for (const item of data.results) {
            const patent = data.patents[item.patent]
            const snippet = item.search_snippet.replace(/<br>/g, '');
            const itemHtml = `
                <div class='patent-entry'>
                <h5><a href='${patent.href}' target='_blank'>
                    ${patent.title}
                    (${patent.pub_id})
                    </a></h5>
                <div class='patent-snippet'>${snippet}</div>
                <div class='patent-description'>${patent.abstract_snippet}</div>
                </div>
            `;
            itemHtmls.push(itemHtml);
        }
        return `
        <div>
            <a href='${data.search.href}' target='_blank'>View on Google Patents</a> (Results may differ)<br/>
            Total Results: ${data.search.total_results}
            <hr/>
            ${itemHtmls.join('\n')}
        </div>
        `
    }

    previewPatents(drugNames) {
        const diseaseNames = this.diseaseNames.getValues();

        const query = {
            drugNames,
            diseaseNames
        }
        assert(drugNames, "Must have some drugs");
        for (const drugNameList of drugNames) {
            assert(drugNameList, "Must have some names for each drug");
        }
        assert(diseaseNames, "Must have some names for disease");

        console.info("Searching on", query);

        const queryStr = JSON.stringify(query);

        const csrf = document.getElementsByName("csrfmiddlewaretoken")[0].value
        const formData = new FormData()
        formData.append('csrfmiddlewaretoken', csrf)
        formData.append('query', queryStr)
        const opts = {
            method: "POST",
            body: formData
        };

        fetch(`/pats/${this.ws}/preview_search/`, opts).then((resp) => {
            if (!resp.ok) {
                resp.text().then((respText) => {
                    console.info("Returned with ", respText);
                    showModal('Error', respText)
                });
                return;

            }
            return resp.json()
        }).then((respJson) => {
            console.info("Returned with ", respJson);
            showModal('Patent Search Results', this.resultsPreviewHtml(respJson));
        });
    }

    makeQuery() {
        const diseaseNames = this.diseaseNames.getValues();
        const drugList = this.drugs.getDrugList();

        const tableNumber = this.searchEl.querySelector('#table-selection select').value;

        const query = {
            drugList,
            diseaseNames,
            tableNumber,
        };
        return query;
    }

    async search() {
        this.searchBtn.setAttribute('disabled', true);
        const query = this.makeQuery();
        console.info("Searching on", query);

        const queryStr = JSON.stringify(query);

        const csrf = document.getElementsByName("csrfmiddlewaretoken")[0].value
        const formData = new FormData()
        formData.append('csrfmiddlewaretoken', csrf)
        formData.append('query', queryStr)
        formData.append('search_btn', true)
        const opts = {
            method: "POST",
            body: formData
        };

        const resp = await fetch(`/pats/${this.ws}/search/`, opts);
        if (!resp.ok) {
            const respText = await resp.text();
            showModal('Error', respText)
            return;
        }
        const respJson = await resp.json();
        const nextUrl = respJson.next_url;
        window.location = nextUrl;
    }


    _saveState() {
        const state = JSON.stringify(this.dumpState());
        localStorage.setItem(this._localStorageKey, state);
    }

    dumpState() {
        return this.makeQuery();
    }

    loadState(data) {
        this._setupInputs(data);
    }
}


function showModal(title, content) {
    if (!window.modalEl) {
        window.modalEl = $("<div class='modal fade' role='dialog'/>");
        document.body.appendChild(window.modalEl[0]);
    }
    window.modalEl[0].innerHTML = `
      <div class="modal-dialog modal-lg" role="document">
        <div class="modal-content">
          <div class="modal-header">
            <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
            <h4 class="modal-title">${title}</h4>
          </div>
          <div class="modal-body">
            ${content}
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-default" data-dismiss="modal">Close</button>
          </div>
        </div><!-- /.modal-content -->
      </div><!-- /.modal-dialog -->
    `;
    window.modalEl.modal('show');
}
