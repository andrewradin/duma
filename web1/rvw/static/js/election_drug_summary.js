


function setupHooks(data, reviewedTargets, prelimReviewedTargets) {
    for (const drug of data) {
        const checkEl = $(`#id_d_${drug.wsa_id}`)[0];
        checkEl.addEventListener('change', () => {
            updateHtml(data, reviewedTargets, prelimReviewedTargets);
        });
    }
}

function wsaIsSelected(wsaId) {
    return $(`#id_d_${wsaId}`).prop('checked');
}

function generateSummaryHtml(data, reviewedTargets, prelimReviewedTargets) {
    const selectedGeneCounts = {}
    let pscrIds = []
    for (const drug of data) {
        if (!wsaIsSelected(drug.wsa_id)) {
            continue;
        }
        pscrIds = drug.prescreen_ids
        for (const target of drug.targets) {
            const gene = target[0];
            if (!(gene in selectedGeneCounts)) {
                selectedGeneCounts[gene] = 0;
            }
            selectedGeneCounts[gene]++;
        }
    }

    const cols = [
        'Drug',
        'Targets',
        `Prescreen Ranks (ID #s: ${pscrIds})`
    ];
    const bodyRows = []
    let selectedCount = 0;
    for (const drug of data) {
        const genes = [];
        const isSelected = wsaIsSelected(drug.wsa_id);
        selectedCount += isSelected;
        for (const target of drug.targets) {
            const gene = target[0];
            const geneHtml = `<a href="${target[1]}">${target[0]}</a>`;
            if (isSelected && (gene in selectedGeneCounts) && selectedGeneCounts[gene] > 1) {
                genes.push(`<b>${geneHtml}</b>`);
            } else {
                genes.push(geneHtml);
            }
        }


        const ranks = [];
        for (let i = 0; i < drug.ranks.length; ++i) {
            if (drug.is_mark_pscr[i]) {
                ranks.push(`<u>${drug.ranks[i]}</u>`);
            } else {
                ranks.push(drug.ranks[i]);
            }
        }

        const row = [
            `<a href="${drug.wsa_href}">${drug.canonical}</a>`,
            genes.join(' '),
            `${ranks.join('')}`,
        ];

        const selStyle = isSelected ? '' : 'background-color: #f3f3f3; color: #888';
        const selClass = isSelected ? 'selected' : 'unselected';
        
        bodyRows.push(`<tr class='${selClass}' style='${selStyle}'><td>${row.join('</td><td>')}</td></tr>`);
    }

    const header = `<tr><th>${cols.join('</th><th>')}</th></tr>`;
    const body = bodyRows.join('');

    const overlapped = [];
    for (const gene in selectedGeneCounts) {
        const count = selectedGeneCounts[gene];
        if (count > 1) {
            overlapped.push(gene);
        }
    }
    const reviewedOverlap = [];
    for (const reviewedGene in reviewedTargets) {
        const curOverlap = [];
        for (const reviewedTarget of reviewedTargets[reviewedGene]) {
            const [name, electionUrl] = reviewedTarget;
            if (selectedGeneCounts[reviewedGene] > 0) {
                curOverlap.push(`<a href='${electionUrl}'>${name}</a>`);
            }
        }
        if (curOverlap.length > 0) {
            reviewedOverlap.push(`${reviewedGene} (${curOverlap})`)
        }
    }
    const prelimReviewedOverlap = [];
    for (const reviewedGene in prelimReviewedTargets) {
        const curOverlap = [];
        for (const reviewedTarget of prelimReviewedTargets[reviewedGene]) {
            const [name, electionUrl] = reviewedTarget;
            if (selectedGeneCounts[reviewedGene] > 0) {
                curOverlap.push(`<a href='${electionUrl}'>${name}</a>`);
            }
        }
        if (curOverlap.length > 0) {
            prelimReviewedOverlap.push(`${reviewedGene} (${curOverlap})`)
        }
    }
    return `
        <b>Drugs Selected:</b> ${selectedCount}/${data.length}<br/>
        <b>Overlapping targets:</b> ${overlapped.join(', ')}<br/>
        <b>Overlap with targets in prelim elections:</b> ${prelimReviewedOverlap.join(', ')}<br/>
        <b>Overlap with targets in final elections:</b> ${reviewedOverlap.join(', ')}<br/>
        <table class='table table-condensed'>
            <thead><tr>${header}</thead>
            <tbody>${body}</tbody>
        </table>
    `;
}

function updateHtml(data, reviewedTargets, prelimReviewedTargets) {
    const el = document.getElementById('drug-summary');
    el.innerHTML = generateSummaryHtml(data, reviewedTargets, prelimReviewedTargets);
}

function setupDrugSummary(data, reviewedTargets, prelimReviewedTargets) {
    setupHooks(data, reviewedTargets, prelimReviewedTargets);
    updateHtml(data, reviewedTargets, prelimReviewedTargets);
}

function setupWhenShown(ws_id, wsa_ids) {
    let loaded = false;
    const wsalist = wsa_ids.join(',')
    $('#sec_drug_summary').on('show.bs.collapse', async () => {
        if (!loaded) {
            loaded = true;
            const resp = await fetch(`/rvw/${ws_id}/election/summary/${wsalist}/`);
            const data = await resp.json();
            setupDrugSummary(
                data.drug_summary_data,
                data.reviewed_targets,
                data.reviewed_targets_prelim);
        }
    })
}
