import * as React from 'react';
import {useState, useEffect, useRef} from 'react';
import * as ReactDOM from 'react-dom';
import {HoverInfo} from './pathway_tree';
import {targetsetPathwayScores} from './pathway_heatmap';

// These come from the page, using the common version we use elsewhere.
//const $ = require('jquery');
//const DataTable = require('datatables.net')(window, $);



function DataTableView({columns, tableData, id, onClick}) {
    const divRef = useRef(null);

    useEffect(() => {
        if (!divRef || !divRef.current) {
            return;
        }

        divRef.current.innerHTML = '';
        const divEl = $(divRef.current);
        const tableEl = $('<table id="' + id + '" class="table table-condensed">');
        divEl.append(tableEl);
        const dataTable = tableEl.DataTable({
            data: tableData,
            columns: columns,
            order: [[1, 'desc']],
        });

        if (onClick) {
            tableEl.on('click', 'tbody td', function() {
                //console.info("Click", this, dataTable.row(this));
                onClick({row: dataTable.row(this).index(), col: this.cellIndex});
            });
        }
        
    });

    return (<div ref={divRef}></div>);
}

export function allDescendants({id, hierarchy, seen, pathTo, curPath}) {
    // There are a few dupes in the tree, where the same reaction/pathway is
    // multiparented.  Remove them.
    seen = seen || {};
    curPath = curPath || [];
    if (seen[id]) {
        return [];
    }
    seen[id] = true;
    pathTo[id] = [...curPath];

    curPath.push(id);
    const descs = [];
    if (id in hierarchy) {
        for (const childId of hierarchy[id]) {
            descs.push(allDescendants({id: childId, hierarchy, seen, pathTo, curPath}));
        }
    }
    curPath.pop();
    return [id].concat(...descs);
}

function PathwayScoreTable({storeData, pathwayId, pathwayData, scorers, callbacks, pathwayFilterer}) {

    // Collect all descendant pathways & scores

    // Contains the path from the current node to each descendant node.
    const pathTo = {};
    const pathwaysUnfiltered = allDescendants({id: pathwayId, hierarchy: pathwayData.hierarchy, pathTo});
    const pathways = pathwaysUnfiltered.filter(pathwayFilterer);
    

    const {pathScorer, protScorer, protAgg} = scorers;

    // Change to rows for pathways, cols to score types

    // Filter to only pathway scores.
    const scores = storeData.scores.filter((score) => 'pwToScore' in score);
    const columns = [
        {'title': 'Pathway'},
        {'title': 'DisAggScore'},
        {'title': 'nProts'},
        {'title': 'Hier'},
    ];
    for (const score of scores) {
        columns.push({'title': score.title});
    }
    const collections = storeData.collections;
    for (const collection of collections) {
        columns.push({'title': collection.title});
    }

    const tableData  = [
    ];

    const collScores = [];
    for (const collection of collections) {
        const prots = collection.targets.map(x => x.uniprot);
        collScores.push(targetsetPathwayScores(prots, pathwayData.protToPathways, pathwayData.protsets, pathways));
    }


    const kMaxHierarchy = 3;
    // Max char-length of a label.
    const kMaxHierLabel = 50;
    let pwIdx = 0;
    for (const pid of pathways) {
        const prots = pathwayData.protsets[pid] || [];
        const name = pathwayData.idToName[pid] || pid;
        const nameEl = `<a>${name}</a>`;

        let hierPath = pathTo[pid];
        // Pop off the first one, it's the currently selected pathway id.
        hierPath.splice(0, 1);
        hierPath = hierPath.map((id) => `<span class='hier-nonlink'>${(pathwayData.idToName[id] || id).substr(0, kMaxHierLabel)}</span>`);
        if (hierPath.length > kMaxHierarchy+1) {
            const extraList = hierPath.splice(kMaxHierarchy);
            const hoverMore = `
                ... +${extraList.length}<span class='hoverable'>
                    <span class='glyphicon glyphicon-info-sign'></span>
                    <span class='hover-text info-hover'>
                        ${extraList.join('<br>')}
                    </span>
                </span>
                `;
            hierPath.push(hoverMore);
        }

        const row = [
            nameEl,
            '', // placeholder for agg score
            prots.length,
            hierPath.join('<br>'),
        ];

        let sum = 0;
        for (const score of scores) {
            const weight = score.weight;
            const pwScore = pathScorer(pid, score);
            sum += weight * pwScore;
            row.push(pwScore.toFixed(3));
        }
        for (const collScore of collScores) {
            row.push(collScore[pwIdx].toFixed(3));
        }

        // Replace the placeholder with the actual agg score value.
        row[1] = sum.toFixed(4);

        tableData.push(row);
        pwIdx += 1;
    }

    // Allow clicking on the pathway name to take you to that pathway in the hierarchy.
    const onClick = ({row, col}) => {
        if (col == 0) {
            const pid = pathways[row];
            callbacks.selectPathwayById(pid);
        }
    };

    return (
        <DataTableView id="PathwayScoreTable" columns={columns} tableData={tableData} onClick={onClick} />
    );
}

function ProteinScoreTable({storeData, pathwayData, pathwayId, scorers, config}) {
    const scores = storeData.scores.filter((score) => !('pwToScore' in score));
    const prots = pathwayData.protsets[pathwayId] || [];
    const tableData  = [
    ];
    const columns = [
        {'title': 'Gene'},
        {'title': 'Mean'},
    ];
    for (const score of scores) {
        columns.push({'title': score.title});
    }
    const {pathScorer, protScorer, protAgg} = scorers;

    for (const prot of prots) {
        const row = [pathwayData.prot2gene[prot] || `(${prot})`];

        // Add a sum column.
        const sumIdx = row.length;
        row.push(0);

        for (const scoreDetails of scores) {
            const protScoreArr = protScorer(prot, scoreDetails, config);
            if (protScoreArr.length == 0) {
                row.push('');
            } else {
                console.assert(protScoreArr.length == 1);
                const protScore = protScoreArr[0];
                row.push(protScore.toFixed(4));
                const weight = scoreDetails.weight;
                row[sumIdx] += weight * protScore;
            }
        }

        // Convert sum to mean.
        row[sumIdx] = (row[sumIdx] / scores.length).toFixed(6);

        tableData.push(row);
    }
    return (
        <DataTableView id="ProteinScoreTable" columns={columns} tableData={tableData} />
    );
}

export function ScoreDetails({storeData, pathwayData, pathwayId, scorers, config, callbacks, pathwayFilterer}) {
    return (
    <div>
        <div className='score-table-panel'>
            <div>
                <h4 style={{'display': 'inline-block'}}>Pathway Scores</h4>&nbsp;
                <HoverInfo>
                    Displays scores for the selected pathway and all of its descendants.
                </HoverInfo>
            </div>
            <PathwayScoreTable storeData={storeData} pathwayId={pathwayId} pathwayData={pathwayData} scorers={scorers} callbacks={callbacks} pathwayFilterer={pathwayFilterer}/>
        </div>
        <div className='score-table-panel'>
        <h4>Protein Scores</h4>
        <ProteinScoreTable storeData={storeData} pathwayId={pathwayId} pathwayData={pathwayData} scorers={scorers} config={config}/>
        </div>
    </div>
    );
}