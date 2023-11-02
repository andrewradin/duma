

function tableToTsv(table, skipCols=[]) {
    const out = []

    for (const row of table.querySelectorAll('tr')) {
        const rowVals = [];
        let colIdx = 0;
        for (const cell of row.querySelectorAll('td, th')) {
            if (skipCols.indexOf(colIdx) == -1) {
                let val = cell.innerText;
                val = val.replace(/[\n\t]/g, '    ');
                rowVals.push(val); 
            }
            colIdx += 1;
        }
        out.push(rowVals.join('\t'));
    }
    return out.join('\n');
}

function downloadContent(content, downloadFilename) {
    var blob = new Blob([content], {type: 'text/tsv'});
    var el = document.createElement('a');
    el.href = window.URL.createObjectURL(blob);
    el.download = downloadFilename;        
    document.body.appendChild(el);
    el.click();        
    document.body.removeChild(el);
}
