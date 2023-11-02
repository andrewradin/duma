


function fromPlotData(el, plotData, decPlaces) {

    let colNames = []
    const data = []
    for (const entry of plotData) {
        let sum = 0;
        for (const y of entry.y) {
            sum += y;
        }

        const row = [entry.name, sum.toFixed(decPlaces)];
        for (const y of entry.y) {
            row.push(y.toFixed(decPlaces));
        }
        colNames = ['Name', 'Sum'].concat(entry.x);
        data.push(row);
    }

    const columns = [];
    for (const colName of colNames) {
        columns.push({title: colName});
    }

    el.DataTable({
        data,
        columns,
        order: [[1, 'desc']],
    });
}

function fromArray(el, data, columnNames) {
    console.info("Setting ", data)
    const columns = [];
    for (const colName of columnNames) {
        columns.push({title: colName});
    }
    el.DataTable({
        data,
        columns,
        order: [[1, 'desc']],
    });
}
