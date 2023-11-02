
import _ from 'lodash';

/**
 * Adds per-column filters to a DataTable.
 * 
 * config: {column #: type_str}
 *      e.g.{
 *             0: 'text',
 *             2: 'text',
 *             3: 'select',
 *             4: 'dropdown',
 *             5: 'range',
 *           }
 * 
 * Implemented types include:
 *    - text:  Text search box, shows only rows where this column contains searched text
 *    - select, dropdown: Multi and single select boxes, useful when there are only a small
 *                        number of values this column can take on.
 *                        Also summarizes the count of each option in the dropdown itself.
 *    - range: Numeric range, can set minimum and/or maximum values to display
 */
export function addDataTableFilters(table, config) {
    function countValues(data) {
        const out = {}
        data.each((x) => {
            x = $(`<span>${x}</span>`).text();
            if (!(x in out)) {
                out[x] = 0;
            }
            out[x] += 1;
        });
        return out;
    }
    const footer = $('<tfoot><tr></tr></tfoot>');
    const footerRow = footer.find('tr');
    const filters = [];
    table.columns().every(function(idx) {
        const col = this;
        const cell = $('<th/>');

        let cfg = config[`${idx}`];


        if (cfg) {
            let content;
            if (cfg == 'text') {
                content = $('<input class="form-control input-sm dt-filter" placeholder="Filter" />');
            } else if (cfg == 'dropdown' || cfg == 'select') {
                const counts = countValues(col.data());
                // This seemed like a good idea, but it makes the UI inconsistent, let's try not for now.
                /*
                if (Object.keys(counts).length <= 1) {
                    cfg = 'dropdown';
                }*/
                let multi = '';
                if (cfg == 'select') {
                    multi = 'multiple';
                }
                content = `<select class="form-control input-sm dt-filter" ${multi}>`;
                if (cfg == 'dropdown') {
                    // Add an 'empty' option for dropdown, so you can remove the filter.
                    content += `<option value=""></option>`;
                }
                for (const [name, count] of _.sortBy(Object.entries(counts), [(x) => -x[1], (x) => x[0]])) {
                    content += `<option value="${name}">${name} (${count})</option>`;
                }
                content += '</select>';
                content = $(content);
            } else if (cfg == 'range') {
                content = $(`
                <input id='min' type="number"class="form-control dt-filter" placeholder="Min" />
                <input id='max' type="number"class="form-control dt-filter" placeholder="Max" />
                    `);
                const minRef = [NaN];
                const maxRef = [NaN];

                // We invoke these on change with "refreshRange=true".  Subsequently
                // it will get invoked for each row of data to filter or not.
                filters.push((data, refreshRange) => {
                    if (refreshRange) {
                        minRef[0] = parseFloat(content.filter('#min').val());
                        maxRef[0] = parseFloat(content.filter('#max').val());
                        return true;
                    }
                    const min = minRef[0];
                    const max = maxRef[0];
                    const val = data[idx];
                    if (!isNaN(min) && val < min) {
                        return false;
                    }
                    if (!isNaN(max) && val > max) {
                        return false;
                    }
                    return true;
                });
            }

            content.on('change keyup clear', function() {
                let val = $(this).val();
                if (cfg == 'text') {
                    col.search(this.value).draw();
                } else if (cfg == 'select') {
                    if (val === null) {
                        col.search('').draw();
                    } else {
                        const escVals = val.map(_.escapeRegExp);
                        const search = '^(' + escVals.join('|') + ')$';
                        // Use regex, not smart search.
                        col.search(search, true, false).draw();
                    }
                } else if (cfg == 'dropdown') {
                    col.search(val, false, false).draw();
                } else if (cfg == 'range') {
                    for (const filter of filters) {
                        filter(null, true);
                    }
                    col.draw();
                }
            });
            cell.append(content);
        }
        footerRow.append(cell);

    });

    $(table.table().node()).append(footer);

    $.fn.dataTable.ext.search.push((settings, data, dataIndex) => {
        if (settings.nTable != table.table().node()) {
            // Global filter, different table.
            return true;
        }
        for (const filter of filters) {
            if (!filter(data, false)) {
                return false;
            }
        }
        return true;
    });
}

/**
 * This function reformats a table to make it more legible.
 * - Adjacent cells in the same column with same content are merged
 * - Cells with too many characters for the # of lines are truncated
 * - Special hover support that handles spanned rows and untruncates
 * 
 * colsToMerge can be used as a list of columns to only merge the specified indices.
 * maxCharsPerRow can be set to 0 to turn off truncation.
 * 
 * The return value should be used as a drawCallback on a datatable.
 */
export function makeDataTableMerger(colsToMerge=null, maxCharsPerRow=20) {

    function drawCallback(settings) {
        var api = this.api();
        var rows = api.rows( {page:'current'} ).nodes();
        const doTruncate = maxCharsPerRow > 0;
        const colCount = api.columns().header().length;
        const inRow = [];
        for (let row = 0; row < rows.length; ++row) {
            inRow.push([]);
        }
        for (let col = 0; col < colCount; ++col) {
            const doMerge = colsToMerge === null || colsToMerge.indexOf(col) != -1;

            let lastVal = null;
            let lastCell = null;
            var span = 1;
            function truncateLast() {
                if (!doTruncate) {
                    return;
                }
                if (lastCell && lastCell.fullContent === undefined) {
                    const maxLength = span * maxCharsPerRow;
                    lastCell.fullContent = lastCell.textContent;
                    if (lastVal.length > maxLength) {
                        lastCell.textContent = lastVal.substr(0, maxLength - 3) + '...';
                        lastCell.style.minWidth = '150px';
                    }
                    lastCell.shortContent = lastCell.textContent;
                }
            }
            for (let row = 0; row < rows.length; ++row) {
                const cell = rows[row].children[col];
                cell.setAttribute('rowspan', 1);
                const cellVal = cell.fullContent || cell.textContent;
                cell.style.display = '';
                if (cellVal === lastVal && doMerge) {
                    span += 1
                    lastCell.setAttribute('rowspan', span);
                    cell.style.display = 'none';
                } else {
                    truncateLast()
                    span = 1;
                    lastCell = cell;
                    lastVal = cellVal;
                    const curRow = row;
                    $(lastCell).unbind('mouseenter mouseleave').hover(function() {
                        const $el = $(this);
                        if (doTruncate) {
                            this.textContent = this.fullContent;
                        }
                        const rowSpan = this.rowSpan;
                        for (let i = 0; i < rowSpan; ++i) {
                            const els = $(inRow[curRow + i]);
                            els.addClass('hover');
                        }
                    }, function() { 
                        $(this).parent().parent().find('td').removeClass('hover')
                        if (doTruncate) {
                            this.textContent = this.shortContent;
                        }
                    });
                }
                inRow[row].push(lastCell);
            }
            truncateLast();
        }
    }
    return drawCallback;
}
