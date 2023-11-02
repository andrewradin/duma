import * as React from 'react';
import {useState, useEffect} from 'react';
import * as ReactDOM from 'react-dom';
import {Plot} from './plotly';
import {allDescendants} from './score_details';
import {kRootNodeId, Hierarchy} from './pathway_data';
import {HoverInfo, JobScoreData} from './pathway_tree';
import {NodePathwayScorer, PathwayScorers} from './pathway_scoring';
import {makeScatterTraces} from './pathway_heatmap';

import * as _ from 'lodash';


// TODO:
// Definte a pathway KT set by taking our normal KT set, generating pathway enrichment,
// and setting a threshold?  threshold %?  manually specifiable?


function ScatterPlot({xScorer, yScorer, ids, xTitle, yTitle, pathwayData, callbacks, cursorPathwayId, hideZeros}) {
    const pathTo = {}
    if (cursorPathwayId != kRootNodeId) {
        allDescendants({
            id:cursorPathwayId,
            hierarchy:pathwayData.hierarchy,
            pathTo
        });
    }

    const scatterData = [];
    for (const id of ids) {
        const pwName = `${pathwayData.idToName[id]} (${id})`;
        const xScore = xScorer(id);
        const yScore = yScorer(id);
        if (hideZeros && (!xScore || !yScore)) {
            continue;
        }
        if (!xScore && !yScore) {
            // If both are 0, then always hide.
            // Otherwise this gets super slow due to tons of 0,0 points.
            continue;
        }

        scatterData.push([xScore, yScore, pwName, id, ''])
    }

    const scatterTraces = makeScatterTraces({scatterData, pathTo});

    const scatterLayout = {
        hovermode: 'closest',
        xaxis: {
            title: xTitle,
        },
        yaxis: {
            title: yTitle,
        },
    };

    const onScatterClick = (e) => {
        if (e.points && e.points.length > 0) {
            const p = e.points[0];
            const colIdx = p.pointIndex;
            const pwId = p.data.pwId[colIdx];
            callbacks.selectPathwayById(pwId);
        }
    };

    return <Plot
            data={scatterTraces}
            layout={scatterLayout}
            useResizeHandler={true}
            onClick={onScatterClick}
            style={{width: '100%', height: '800px'}}
        />;
}

export function PathwayCompare({storeData, pathwayData, scorers, callbacks, cursorPathwayId}) {
    let initialState;
    try {
        initialState = JSON.parse(localStorage.getItem('PathwayCompareState')) || {hideZeroes: true};
    } catch (e) {
        initialState = {hideZeros: true};
    }
    const [state, setReactState] = useState(initialState);
    const updState = (changes) => {
        const newState = {...state, ...changes};
        localStorage.setItem('PathwayCompareState', JSON.stringify(newState));
        setReactState(newState);
    };


    // Within each score, have pwToScore, minscore, maxScore.
    const scores = storeData.scores;
    const collections = storeData.collections;
    const jids = [];
    const ws = []
    const {pathScorer, protScorer, protAgg} = scorers;

    let xScore = null;
    let yScore = null;
    let xTitle = '';
    let yTitle = '';

    const scoreChoices= [<option value={''}></option>];
    const groups = {}
    for (const score of scores) {
        scoreChoices.push(<option value={score.jobId}>{score.title}</option>);
        if (score.jobId == state.xScoreId) {
            xScore = (x) => pathScorer(x, score);
            xTitle = score.title;
        }
        if (score.jobId == state.yScoreId) {
            yScore = (x) => pathScorer(x, score);
            yTitle = score.title;
        }

        const wskey = `ws${score.wsId}`;
        if (!(wskey in groups)) {
            groups[wskey] = {
                title: wskey + ' Sum',
                scores: [score],
            };
        } else {
            groups[wskey].scores.push(score);
        }

        const wfGroupKey = score.groupLabel;
        if (wfGroupKey) {
            if (!(wfGroupKey in groups)) {
                groups[wfGroupKey] = {
                    title: wfGroupKey + ' Sum',
                    scores: [score],
                };
            } else {
                groups[wfGroupKey].scores.push(score);
            }
        }
    }

    for (const groupId in groups) {
        const groupData = groups[groupId];
        scoreChoices.push(<option value={groupId}>{groupData.title}</option>);
        function scoreSum(pwId) {
            let out = 0;
            for (const score of groupData.scores) {
                out += pathScorer(pwId, score);
            }
            return out;
        }
        if (groupId == state.xScoreId) {
            xScore = scoreSum;
            xTitle = groupId;
        }
        if (groupId == state.yScoreId) {
            yScore = scoreSum;
            yTitle = groupId;
        }
    }

    let plot = null;
    if (xScore && yScore) {
        //const pathwaysToPlot = _.uniq(_.concat(Object.keys(xScore.pwToScore), Object.keys(yScore.pwToScore)));
        const pathwaysToPlot = Object.keys(pathwayData.idToName);
        plot = <ScatterPlot
            xScorer={xScore}
            yScorer={yScore}
            ids={pathwaysToPlot}
            xTitle={xTitle}
            yTitle={yTitle}
            pathwayData={pathwayData}
            callbacks={callbacks}
            cursorPathwayId={cursorPathwayId}
            hideZeros={state.hideZeros}
            />;
    }


    const controlStyle = {
        marginLeft: "0.25rem",
        marginRight: "0.25rem",
        padding: "0.5rem",
    };
    return (
        <>
        <div className='row form-inline'>
            <span style={controlStyle}>
                X: <select value={state.xScoreId} onChange={(e) => updState({xScoreId: e.target.value})} >
                    {scoreChoices}
                </select>
            </span>
            <span style={controlStyle}>
                Y: <select value={state.yScoreId} onChange={(e) => updState({yScoreId: e.target.value})} >
                    {scoreChoices}
                </select>
            </span>
            <span style={controlStyle}>
                Hide Zeros: <input type='checkbox' checked={state.hideZeros} onChange={(e) => updState({hideZeros: e.target.checked})}></input>
            </span>
        </div>
        <div className='row'>
            {plot}
        </div>
        </>
    );
};

