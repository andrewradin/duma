import * as React from 'react';
import {useState, useEffect} from 'react';
import * as ReactDOM from 'react-dom';
import {Plot} from './plotly';
import {allDescendants} from './score_details';
import {kRootNodeId, Hierarchy} from './pathway_data';
import {HoverInfo, JobScoreData, kColorList} from './pathway_tree';
import {NodePathwayScorer, PathwayScorers} from './pathway_scoring';

import * as _ from 'lodash';


const kPathwayMean = "Pathway Mean";

function arrayMean(arr: number[]) {
    let out = 0;
    for (const el of arr) {
        out += el;
    }
    return out / arr.length;
}



export function targetsetPathwayScores(targets, protToPathways, pathwayToProts, allPathways) {
    // For a target list, go through each target.
    // Grab the set of pathways.
    const pathwayCounter: {[key: string]: number} = {}
    for (const target of targets) {
        const pathways = protToPathways[target];
        if (pathways) {
            for (const pathway of pathways) {
                if (!(pathway in pathwayCounter)) {
                    pathwayCounter[pathway] = 0;
                }
                pathwayCounter[pathway] += 1;
            }
        }
    }

    const out = []
    for (const pathway of allPathways) {
        if (pathway in pathwayCounter) {
            const pathwaySize = pathwayToProts[pathway].length;
            out.push((pathwayCounter[pathway] / pathwaySize));
        } else {
            out.push(0);
        }
    }
    return out;
}


function allScoredPathways(scores: JobScoreData[], collections, pwScorer: NodePathwayScorer) {
    const out: {[key: string]: boolean} = {}
    for (const score of scores) {
        if (score.pwToScore) {
            for (const pwId in score.pwToScore) {
                const scoreVal = pwScorer(pwId, score);
                if (scoreVal > 0) {
                    out[pwId] = true;
                }
            }
        }
    }
    for (const col of collections) {
        for (const [pwId, pwInfo] of Object.entries(col.pathways)) {
            if (pwInfo['type'] == 'pathway') {
                out[pwId] = true;
            }
        }
    }
    return Object.keys(out);
}

function makeHeatmap(scores, collections, pathwayData, storeData, {removeReactions, scorers, plotType, callbacks, sortBy, cursorPathwayId, numClusters, clusterType, fixedScale}) {
    console.info("Creating heatmap");
    // Filter out non-pathway scores.
    scores = scores.filter((score) => {
        return 'pwToScore' in score;
    });

    const pwScorer = scorers.pathScorer;
    const allPathways = allScoredPathways(scores, collections, pwScorer);

    const pathwayToProts = pathwayData.protsets;
    const protToPathways = {}
    for (const [pathway, prots] of Object.entries(pathwayToProts)) {
        for (const prot of prots) {
            if (!(prot in protToPathways)) {
                protToPathways[prot] = [];
            }
            protToPathways[prot].push(pathway);
        }
    }

    let tracedata = []
    const rowNames = []; // Score names.

    for (const score of scores) {
        rowNames.push(score.title);
    }

    const filterPathways = (pw) => {
        if (!(pw in pathwayData.pathways)) {
            // We don't have data on some legacy pathways, just pass through.
            return true;
        }
        const type = pathwayData.pathways[pw].type;

        if (type == 'event' && removeReactions) {
            return false;
        }
        if (!(pw in pathwayToProts) || pathwayToProts[pw].length == 0) {
            return false;
        }
        return true;
    }
    const goodPathways = allPathways.filter(filterPathways);

    for (const pathway of goodPathways) {
        const pwData = []
        for (const score of scores) {
            pwData.push(pwScorer(pathway, score));
        }

        tracedata.push({
            points: pwData,
            pathway: pathway
        });
    }

    if (scores.length > 0) {
        for (const pwData of tracedata) {
            const mean = arrayMean(pwData.points);
            pwData.points.push(mean);
        }
        rowNames.push(kPathwayMean);
    }

    const meanIdx = rowNames.length - 1;

    const pathTo = {}
    if (cursorPathwayId != kRootNodeId) {
        allDescendants({
            id:cursorPathwayId,
            hierarchy:pathwayData.hierarchy,
            pathTo
        });
    }

    const scatterData = [];
    let relevantPathways: {[key:string]: any} = {}; 
    let colIdx = 0;
    for (const collection of collections) {
        const prots = collection.targets.map(x => x.uniprot);
        const collScores = targetsetPathwayScores(prots, protToPathways, pathwayToProts, goodPathways);
        console.assert(collScores.length == tracedata.length, "Should both be scores for all pathways");
        for (let i = 0; i < collScores.length; ++i) {
            const pwData = tracedata[i];
            const collScore = collScores[i];
            pwData.points.push(collScore);

            const meanPwScore = pwData.points[meanIdx];

            if (collScore > 0 && meanPwScore > 0) {
                const pwName = `${pathwayData.idToName[pwData.pathway]} (${pwData.pathway})`;
                scatterData.push([meanPwScore, collScore, pwName, pwData.pathway, collection.title]);

                if (clusterType == 'count') {
                    relevantPathways[pwData.pathway] = 1;
                } else if (clusterType == 'pwScore') {
                    relevantPathways[pwData.pathway] = meanPwScore;
                } else if (clusterType == 'molScore') {
                    relevantPathways[pwData.pathway] = collScore;
                } else {
                    console.warn("Unexpected clusterType", clusterType);
                }
            }
        }
    }

    if (numClusters > 1) {
        const clustering = makeClusters({
            relevantPathways,
            hierarchy: pathwayData.hierarchy,
            numSplits: numClusters,
            pathwayId: kRootNodeId,
        });
        assignClusters(clustering,
                    relevantPathways,
                    pathwayData.hierarchy,
                    kRootNodeId,
                    [],
                    "",
                    pathwayData.idToName,
                    );
    } else {
        relevantPathways = null;
    }

    const collNameToColor = {};
    collections.forEach((collection, idx) => {
        collNameToColor[collection.title] = kColorList[idx % kColorList.length];
    });

    const scatterTraces = makeScatterTraces({scatterData, pathTo, collNameToColor, clustering:relevantPathways});

    // Sort pwData, and create colNames.
    const colNames = []; // Pathways
    const zsT = [];

    const sortIdx = rowNames.indexOf(sortBy);
    tracedata = _.sortBy(tracedata, x => -x.points[sortIdx]);

    for (const pwData of tracedata) {
        //const pwName = pwData.pathway;
        const pwName = `${pathwayData.idToName[pwData.pathway]} (${pwData.pathway})`;
        colNames.push(pwName);
        zsT.push(pwData.points);
    }

    function transpose(mat) {
        if (mat.length == 0) {
            return [];
        }
        return mat[0].map((col, i) => mat.map(row => row[i]));
    }
    const zs = transpose(zsT);

    const data = [{
            x: colNames,
            y: rowNames,
            z: zs,
            type: 'heatmap',
            colorscale: 'Viridis',
            colorbar: {
                x: -0.08,
            }
        }];
    const layout = {
        yaxis: {
            side: 'right',
            automargin: true,
            tickangle: -45,
        },
        xaxis: {
            tickangle: 45,
        },
        autosize: true,
    };
    
    const scatterLayout = {
        hovermode: 'closest',
        xaxis: {
            title: 'Mean Pathway Score',
        },
        yaxis: {
            title: 'Molecule/MoA Score',
        },
    };

    if (fixedScale) {
        scatterLayout.xaxis.range = [0, 1];
        scatterLayout.yaxis.range = [0, 1];
    }

    const onClick = (e) => {
        if (e.points && e.points.length > 0) {
            const p = e.points[0];
            const colIdx = p.pointIndex[1];
            const pwId = tracedata[colIdx].pathway;
            callbacks.selectPathwayById(pwId);
        }
    });

    const onScatterClick = (e) => {
        if (e.points && e.points.length > 0) {
            const p = e.points[0];
            const colIdx = p.pointIndex;
            const pwId = p.data.pwId[colIdx];
            callbacks.selectPathwayById(pwId);
        }
    };

    let plot;
    if (plotType == 'scatter') {
        plot = (<Plot
            data={scatterTraces}
            layout={scatterLayout}
            useResizeHandler={true}
            onClick={onScatterClick}
            style={{width: '100%', height: '800px'}}
        />);
    } else if (plotType == 'heatmap') {
        plot = (<Plot
            data={data}
            layout={layout}
            useResizeHandler={true}
            onClick={onClick}
            style={{width: '100%', height: '800px'}}
        />);
    }

    return {plot, rowNames}
}

interface ClusterParms {
    relevantPathways: {[key: string]: any};
    hierSums: {[key: string]: number};
    hierarchy: Hierarchy;
    numSplits: number;
    pathwayId: string;
}
/**
 * On input relevantPathways provides pathway weights (depending on cluster type), and on output is reassigned
 * with the cluster pathway root for each pathway.
 */
function makeClusters({relevantPathways, hierarchy, numSplits, pathwayId, hierSums={}}: ClusterParms): string[] {
    const kOther = '[other]';
    let current = [kOther];
    const assignedSplits: {[key: string]: number} = {};
    assignedSplits[kOther] = 1;

    if (_.isEmpty(hierSums)) {
        hierSums = {};
        function computeHierSums(pid: string): number {
            if (pid in hierSums) {
                return hierSums[pid];
            }
            let sum = 0;
            if (pid in relevantPathways) {
                sum += relevantPathways[pid];
            }
            if (pid in hierarchy) {
                for (const childId of hierarchy[pid]) {
                    sum += computeHierSums(childId);
                }
            }
            hierSums[pid] = sum;
            return sum;
        }
        computeHierSums(pathwayId);
    }

    const otherIds = [...hierarchy[pathwayId]];

    const computeTop = () => {
        return _.maxBy(current, (id: string) => {
            if (id == kOther) {
                let sum = 0;
                for (const id of otherIds) {
                    sum += hierSums[id];
                }
                return sum / assignedSplits[id];
            } else {
                return hierSums[id] / assignedSplits[id];
            }
        });
    };
    const splitOther = () => {
        const splitOut = _.maxBy(otherIds, (id: string) => hierSums[id]);
        current.push(splitOut);
        _.remove(otherIds, (id: string) => id == splitOut);
        assignedSplits[splitOut] = 1;
    };
    for (let i = 1; i < numSplits; ++i) {
        const topId = computeTop();
        if (topId == kOther) {
            splitOther();
        } else {
            assignedSplits[topId] += 1;
        }
    }

    const out: string[] = [];
    for (const group of current) {
        const curSplits = assignedSplits[group];
        if (group == kOther) {
            out.push(pathwayId);
        } else if (curSplits == 1) {
            out.push(group);
        } else {
            out.push(...makeClusters({relevantPathways, hierSums, hierarchy, numSplits:curSplits, pathwayId:group}));
        }
    }

    return out;
}

function assignClusters(clusterNodes: string[],
                        relevantPathways: {[key:string]:any},
                        hierarchy: Hierarchy,
                        pathwayId: string,
                        curPath: string[],
                        curRoot: string,
                        idToName: {[key:string]:string}, 
                        ) {
    curPath.push(pathwayId);
    if (clusterNodes.indexOf(pathwayId) != -1) {
        const [root, ...nonroot] = curPath;
        curRoot = nonroot.map((x) => idToName[x] || x).join(' > ');
        if (curRoot == '') {
            curRoot = '[Other]';
        }
    }
    relevantPathways[pathwayId] = curRoot;

    if (pathwayId in hierarchy) {
        for (const childId of hierarchy[pathwayId]) {
            assignClusters(clusterNodes, relevantPathways, hierarchy, childId, curPath, curRoot, idToName);
        }
    }
    curPath.pop();

}


export function makeScatterTraces({scatterData, pathTo, clustering, collNameToColor}) {
    collNameToColor = collNameToColor || {};
    const out = [];
    const grouped = _.groupBy(scatterData, (data) => {
        const [x, y, name, id, coll] = data;

        if (clustering) {
            return clustering[id];
        } else {
            return coll;
        }
    });
    for (const [name, data] of _.sortBy(Object.entries(grouped))) {
        const [x, y, pwName, pwId, collName] = _.unzip(data);
        const borderArr = pwId.map((x) => x in pathTo ? 2 : 0);
        const trace = {
            x: x,
            y: y,
            text: pwName,
            name: name,
            type: 'scattergl',
            mode: 'markers',
            pwId: pwId,
            marker: {
                line: {
                    width: borderArr,
                    // Has to show up well against the background and default colors.
                    // Red works well for most things except red; we could also manually select
                    // colors here and pick something different when the marker is red.
                    color: '#ef0000',
                },
            },
        };
        if (name in collNameToColor) {
            trace.marker.color = collNameToColor[name];
        }
        out.push(trace);
    }
    return out;
}

// It's not obvious that this is saving anything anymore,
// I suspect we're rerendering every time with all these checks.
function heatmapPropEqual(prevProps, nextProps) {
    const prev = prevProps.storeData;
    const next = nextProps.storeData;

    return (prev.scores == next.scores && 
            prev.collections == next.collections &&
            prevProps.plotType == nextProps.plotType &&
            prevProps.cursorPathwayId == nextProps.cursorPathwayId &&
            prevProps.scorers == nextProps.scorers);
}

// Note that the memo here is using the propEqual func above.
export const PathwayHeatmap = React.memo(({storeData, pathwayData, plotType, scorers, callbacks, cursorPathwayId}) => {
    const initialState = {
        removeReactions: false,
        fixedScale: false,
        sortyBy: kPathwayMean,
        numClusters: 3,
        clusterType: 'count',
    };

    // We save the state of the scatter plot settings across reloads / tab switches.
    // If we don't care about reloads, could alternatively extract the state to the container component,
    // or have the container persist the object even when not visible.
    try {
        // Update the initial state with saved state.
        Object.assign(initialState, JSON.parse(localStorage.getItem('PathwayHeatmapState')));
        console.info("Loaded state, now at ", initialState);
    } catch (e) {
        console.info("No heatmap state to load", e);
    }


    const [state, setReactState] = useState(initialState);

    const {removeReactions, fixedScale, sortBy, numClusters, clusterType} = state;

    const setState = (changes) => {
        const newState = {...state, ...changes};
        localStorage.setItem('PathwayHeatmapState', JSON.stringify(newState));
        setReactState(newState);
    };
    const kClusterTypes = ['count', 'pwScore', 'molScore'];

    // Within each score, have pwToScore, minscore, maxScore.
    const scores = storeData.scores;
    const collections = storeData.collections;
    const jids = [];
    const ws = []

    const onSetRemoveReactions = (e) => setState({removeReactions: e.target.checked});
    const onSetFixedScale = (e) => setState({fixedScale: e.target.checked});

    const controlStyle = {
        marginLeft: "0.25rem",
        marginRight: "0.25rem",
        padding: "0.5rem",
    };

    const plotData = makeHeatmap(scores, collections, pathwayData, storeData, {removeReactions, plotType, callbacks, sortBy, cursorPathwayId, numClusters, scorers, clusterType, fixedScale});
    console.info("Heatmap created");
    const {plot, rowNames} = plotData;
    return (
        <>
        <div className='row form-inline'>
            <span style={controlStyle}>
                Remove Reactions: <input type='checkbox' checked={removeReactions} onChange={onSetRemoveReactions} />
            </span>
            { plotType == 'heatmap' ?  (
                // Heatmap-only controls
                <span style={controlStyle}>
                    Sort: 
                    <select value={sortBy} onChange={(e) => setState({sortBy: e.target.value})}>
                        { rowNames.map((x) => <option value={x}>{x}</option>); }
                    </select>
                </span>
                ) : (
                // Scatter-only controls.
                <>
                <span style={controlStyle}>
                    Clusters or Molecules: 
                            <HoverInfo>
                                The hierarchy will be split into N clusters, attempting to roughly balance
                                the weight sum of pathways in the clusters.<br/>
                                A size of 0 or 1 will instead cluster by molecule/target.
                            </HoverInfo>
                    <input type='number' value={numClusters} onChange={(e) => setState({numClusters: e.target.value})} />
                </span>
                <span style={controlStyle}>
                    ClusterType: 
                            <HoverInfo>
                                count gives all pathways equal weight<br/>
                                pwScore weights pathways by their mean pathway score<br/>
                                molScore weights pathways by their collection/target/molecule score
                            </HoverInfo>
                    <select value={clusterType} onChange={(e) => setState({clusterType: e.target.value})}>
                        { kClusterTypes.map((x) => <option value={x}>{x}</option>); }
                    </select>
                </span>
                <span style={controlStyle}>
                    Fixed Scale: 
                            <HoverInfo>
                                Fixes the x and y axes to range of [0, 1].
                            </HoverInfo>
                    <input type='checkbox' checked={fixedScale} onChange={onSetFixedScale} />
                </span>
                </>
                )
            }
        </div>
        <div className='row'>
            {plot}
        </div>
        </>
    );
}, heatmapPropEqual);

