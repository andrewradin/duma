import * as React from 'react';
import {useState, useEffect} from 'react';
import * as ReactDOM from 'react-dom';

import {Treebeard, decorators} from 'react-treebeard';

import Freezer from 'freezer-js';

import {PathwayData, PathwayNode, kRootNodeId, fetchPathwayData} from './pathway_data';
import {MolSearch, ProtSearch, MolWithTargets} from './mol_search';
import {GeneProt} from './gene_prot';
import {DropdownButton} from './dropdown_button';
import {GlfPicker, ProtScorePicker} from './pickers';
import {PlotlyPlot} from './plotly';
import {PathwayHeatmap} from './pathway_heatmap';
import {PathwayCompare} from './pathway_compare';
import {ScoreDetails} from './score_details';
import {PathwayFilters} from './pathway_filters';

import {AggScoreMarker, CollectionMarker, ScoreMarker} from './score_vis';
import * as Score from './pathway_scoring';

import _ from 'lodash';


/**
 * React Element displaying a glyphicon info sign which, when hovered, displays any child content.
 */
export function HoverInfo({children, className}: any) {
    return (
        <span className='hoverable'>
            <span className='glyphicon glyphicon-info-sign'></span>
            <span className={'hover-text info-hover ' + (className || '')}>
                {children}
            </span>
        </span>
    );
}


function urlDecode(url: string) {
    return decodeURIComponent(url.replace(/\+/g, ' '));
}

const loadInitFromHash = () => {
    const hash = window.location.hash;
    if (hash.length > 1) {
        const initData = JSON.parse(urlDecode(hash.substring(1)));
        return initData;
    } else {
        return {};
    }
}


/**
 * Saves the state into session storage.  This helps retain state on refreshes, which is mostly
 * useful for developing (users probably aren't refreshing as much).
 * If you have a lot of scores loaded in, it will probably exceed session storage limits, so you can't
 * count on this to have the entire state in it, more useful for toy examples.
 */
const saveSession = (data: any) => {
    setTimeout(() => {
        const state = JSON.stringify(data)
        sessionStorage.setItem('state', state);
    }, 1000);
}

/**
 * Loads the saved state from above.
 */
const loadSession = () => {
    const state = sessionStorage.getItem('state');
    if (state) {
        return JSON.parse(state);
    } else {
        return null;
    }
}


/**
 * Fills 'out' with all nodeIds in 'hierarchy' that are ancestors of 'idToFind', starting at 'curId'.
 */
function findPathwayAncestry({idToFind, hierarchy, curId, out}) {
    if (!curId) {
        curId = kRootNodeId;
    }
    if (curId == idToFind) {
        out[curId] = true;
        return true;
    }

    if (curId in hierarchy) {
        for (const childId of hierarchy[curId]) {
            if (findPathwayAncestry({curId: childId, hierarchy, idToFind, out})) {
                out[curId] = true;
                return true;
            }
        }
    }

    return false;
}

/**
 * Creates a 'dedupe map' from a hierarchy and protsets.
 * 
 * Outputs a mapping for each nodeId to a canonical nodeId with the same set of prots (could be itself), such
 * that all nodes with the same protsets get the same singular canonical nodeId.
 * 
 * The id picked as 'Canonical' is assigned arbitrarily (first one encountered).
 * 
 * siblingsOnly restricts this to only 'merge' nodes if they're siblings of each other.
 */
function makeDedupeMap({protsets, hierarchy, id, outMap, protsetMap, siblingsOnly}) {
    outMap = outMap || {};
    protsetMap = protsetMap || {};

    const prots = protsets[id];
    const mapping = protsetMap[prots];
    if (mapping) {
        outMap[id] = mapping;
    } else {
        protsetMap[prots] = id;
    }

    if (id in hierarchy) {
        const childProtsetMap = siblingsOnly ? {} : protsetMap;
        for (const childId of hierarchy[id]) {
            makeDedupeMap({id:childId, protsets, hierarchy, outMap, protsetMap:childProtsetMap, siblingsOnly});
        }
    }

    return outMap;
}


// Matches the plotly color list.
export const kColorList = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#e6194b', '#bcbd22', '#3cb44b',
    '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
    '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
    '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
    ];


/**
 * CSS styling for the 'Clear' <button />
 */
const clearStyle = {
    color: '#dcc',
    fontSize: '90%',
    cursor: 'pointer',
    marginLeft: '2rem',
    textDecoration: 'underline',
}


/**
 * React Element
 * The Molecule Picker widget, including both the search field and the selectors.
 */
const MoleculePicker = ({pathwaysData, store}) => {
    const onMolSelected = (molData) => {
        const targets = molData.targets.map((x) => {
            return {
                uniprot: x[0],
                gene: x[1],
                direction: x[2],
            };
        });

        addTargetCollection(pathwaysData, store, molData.name, targets);
    };
    const classes = "btn btn-default";

    const drugsetMols = async (ws_id, molset_id) => {
        const resp = await fetch(`/api/molset/${ws_id}/${molset_id}`);
        const respData = await resp.json();
        const mols = respData.data;
        return mols.map((x) => ({
            'label': <span><MolWithTargets molData={x} /></span>,
            'data': x,
        }));
    };

    const wsDrugsets = async (ws_id) => {
        const resp = await fetch(`/api/ws_molsets/${ws_id}/`);
        const respData = await resp.json();
        const molsets = respData.data;
        return molsets.map((x) => ({
            'label': x[1],
            'items': drugsetMols.bind(this, ws_id, x[0]),
        }));
    };

    const wsItems = async () => {
        const resp = await fetch('/api/list_workspaces/?active_only=true');
        const respData = await resp.json();
        const wses = respData.data;

        return wses.map((x) => ({
            'label': x.name,
            'items': wsDrugsets.bind(this, x.id),
        }));
    };
    return (
    <>
        <span className='col-sm-4'>
            <MolSearch onSelected={onMolSelected}/>
        </span>
        <span className='col-sm-2'>
            <DropdownButton
                text='WS Molecules'
                items={wsItems}
                classNames={classes}
                onClick={item => onMolSelected(item.data)}
            />
        </span>
    </>
    )
};

/**
 * Adds the given "TargetCollection"  (i.e. molecule, moa, or selected gene(s)) to our selected set.
 */
async function addTargetCollection(pathwaysData, store, title, targets) {
    let loadingEl;
    if (typeof targets == 'function') {
        store = store.push({title: title, targets: [], pathways: {}, loading: true});
        loadingEl = store[store.length - 1];
        targets = await targets();
    }
    const prots = targets.map((x) => x.uniprot);

    const idToNode = {};
    for (const prot of prots) {
        const pathways = pathwaysData.protToPathways[prot];
        if (pathways) {
            for (const pathwayId of pathways) {
                idToNode[pathwayId] = {
                    id: pathwayId,
                    name: pathwaysData.idToName[pathwayId],
                    type: 'pathway',
                }
            }
        }
        idToNode[prot] = {
            id: prot,
            name: prot,
            type: 'prot',
        };
    }

    if (loadingEl) {
        loadingEl.set({targets: targets, pathways: idToNode, loading: false});
    } else {
        store.push({title: title, targets: targets, pathways: idToNode});
    }
}


/**
 * React Element
 * Similar to MoleculePicker above, the UI for selecting Genes to add, both the search box
 * and protset selector.
 */
const GenePicker = ({pathwaysData, store}) => {
    const onProtSelected = ([prot, gene]) => {
        const protData = {
            uniprot:prot,
            gene,
            name: gene,
            direction: 0,
        }
        addTargetCollection(pathwaysData, store, protData.name, [protData]);
    };

    const onProtSetSelected =  async([[psId, psName], wsId, wsName]) => {
        const title = wsName + ' ' + psName;
        const protData = async () => {
            const resp = await fetch(`/api/protset/${wsId}/${psId}/`);
            const respData = await resp.json();
            const prots = respData.data;
            return prots.map((x) => ({
                    uniprot: x[0],
                    gene: x[1],
                    direction: 0
                }));
        };
        addTargetCollection(pathwaysData, store, title, protData);

    };

    const classes = "btn btn-default";

    const wsProtsets = async (wsId, wsName) => {
        const resp = await fetch(`/api/ws_protsets/${wsId}/`);
        const respData = await resp.json();
        const protsets = respData.data;
        return protsets.map((x) => ({
            'label': x[1],
            'data': [x, wsId, wsName],
        }));
    };

    const wsItems = async () => {
        const resp = await fetch('/api/list_workspaces/?active_only=true');
        const respData = await resp.json();
        const wses = respData.data;

        return wses.map((x) => ({
            'label': x.name,
            'items': wsProtsets.bind(this, x.id, x.name),
        }));
    };

    return (
    <>
        <span className='col-sm-1'>
            <DropdownButton
                text='Prot Sets'
                items={wsItems}
                classNames={classes}
                onClick={item => onProtSetSelected(item.data)}
            />
        </span>
        <span className='col-sm-4'>
            <ProtSearch onSelected={onProtSelected}/>
        </span>
    </>
    )
};


/**
 * React Element displaying the info box describing a TargetCollection (e.g. mol/moa/gene(s)).
 */
const TargetCollection = ({title, targets, index, onRemove, addIndirect, pathways, loading}) => {
    const outerStyle = {
        display: 'inline-block',
        padding: '0.5rem',
        margin: '0.5rem',
        backgroundColor: '#f8f9fa',
        maxWidth: '300px',
        border: '1px solid #aac',
        borderRadius: '4px',
    };
    const style = {
        backgroundColor: kColorList[index % kColorList.length],
        width: '10px',
        height: '1rem',
        display: 'inline-block',
        marginRight: '0.5rem',
    };
    const xStyle = {
        cursor: 'pointer',
        float: 'right',
        color: '#a24',
    }
    const addIndirectStyle = {
        display: 'block',
    }

    let content;
    if (loading) {
        content = <span className='loader' />;
    } else {
        content = (<>
            {Object.keys(pathways).length - 1} Pathways/Reactions<br/>
                {targets.map((target, index) => (
                    <>
                        <GeneProt
                            gene={target.gene}
                            prot={target.uniprot}
                            direction={target.direction} />
                    <span> </span>
                    </>
                ))}
            <button className='btn btn-xs btn-default' style={addIndirectStyle} onClick={addIndirect}>
                Add Indirect
                <HoverInfo>
                    Adds a new "Target Collection" with all of the indirect targets (via PPI) of this target collection.<br/>
                    Best when starting with only a small number of targets.
                </HoverInfo>
            </button>
            </>);
    }
    // -1 on pathways length because the list includes itself.
    return (
        <div style={outerStyle}>
            <span style={style}> </span>
            <b>{title} <a onClick={onRemove} style={xStyle}>X</a> </b><br/>
            {content}
        </div>
    );
};


/**
 * React Element grouping the TargetCollection pickers (MoleculePicker, GenePicker) and the display
 * of which TargetCollections are currently enabled.
 */
const TargetPicker = ({pathwaysData, collections}) => {
    const collectionEls = collections.map((data, index) => {
        const onRemove = () => {
            collections.splice(index, 1);
        };
        const addIndirect = async () => {
            const prots = _.map(data.targets, (x) => x.uniprot);
            // Fetch indirects.
            const resp = await fetch(`/api/indirect_targets/${prots.join(',')}`);
            const respData = await resp.json();
            const indTargets = respData.targets;
            // Add new target collection.
            addTargetCollection(pathwaysData, collections, data.title + ' Indirect', indTargets);
        }
        return (<TargetCollection onRemove={onRemove} addIndirect={addIndirect} index={index} {...data} />
    });
    function clearTargets() {
        collections.splice(0, collections.length);
    }
    return (
        <div className='panel panel-primary'>
            <div className='panel-heading'>
                Targets
                <HoverInfo>
                    Add "Target Collections" to show up in the displays below.<br/>
                    These will display a filled-in color-coded rectangle to the left of each pathway if any of the targets are found within the pathway
                </HoverInfo>
                <a style={clearStyle} onClick={clearTargets}>Clear</a>
            </div>
            <div className='panel-body'>
                <div className="row">
                    <MoleculePicker pathwaysData={pathwaysData} store={collections}/>
                    <GenePicker pathwaysData={pathwaysData} store={collections}/>
                </div>
                <div className="row">
                    {collectionEls}
                </div>
            </div>
        </div>
    )
};

/**
 * React Element displaying the info box describing a protein or pathway score.
 * Equivalent to the TargetCollection one above, but for selected scores instead of selected targets.
 */
const ScoreCard = ({index, title, header, pwToScore, protToScore, weight, weights, onRemove}) => {
    const outerStyle = {
        display: 'inline-block',
        padding: '0.5rem',
        margin: '0.5rem',
        backgroundColor: '#f8f9fa',
        maxWidth: '300px',
        border: '1px solid #aac',
        borderRadius: '4px',
    };
    const style = {
        backgroundColor: kColorList[index % kColorList.length],
        width: '10px',
        height: '1rem',
        display: 'inline-block',
        marginRight: '0.5rem',
    };
    const xStyle = {
        cursor: 'pointer',
        float: 'right',
        color: '#a24',
    }
    const scoreStyle = {
        border: '1px solid #aaa',
        padding: '0.25rem',
        margin: '0.25rem',
        fontSize: '80%',
    };
    let descr = '';
    let subcontent = '';
    if (pwToScore) {
        descr = `${Object.keys(pwToScore).length} Pathways Scores`;
    } else if (protToScore) {
        let nonZero = 0;
        for (const score of Object.values(protToScore)) {
            if (score > 0) {
                nonZero += 1;
            }
        }
        descr = `${Object.keys(protToScore).length} (${nonZero} nonzero) Protein Scores`;

    } else {
    }

    if (weights && weight !== undefined) {
        // Everything has a weight, but it's a default=1 unless there are actually input weights.
        subcontent += " (weight: " + weight.toFixed(2) + ")";
    }
    return (
        <div style={outerStyle}>
            <span style={style}> </span>
            <b>{title} <a onClick={onRemove} style={xStyle}>X</a> </b><br/>
            {descr}<br/>
            {subcontent}
        </div>
    );
};

export interface JobScoreData {
    header: string[],
    minScore: number,
    maxScore: number,
    title: string,
    wsId: string,
    jobId: string,
    weights: number[],
    weight: number,

    // -- ProtScore-only fields
    protToRank?: {[key: string]: number},
    // Unnormalized protein scores for this job.
    protToScore?: {[key: string]: number},
    norm?: string,


    // -- PwScore-only fields
    minQScore?: number,
    maxQScore?: number,
    pwToScore?: {[key: string]: number[]},
    groupLabel?: string,

}

async function fetchScores(wsId, jobId, codes) {
    let query = '';
    if (codes) {
        query = `codes=${codes.join(',')}`;
    }
    const resp = await fetch(`/api/fetch_scores/${wsId}/${jobId}/?${query}`);
    const respData = await resp.json()
    return respData
}

interface ScorePickerParms {
    // Technically this is a freezerjs JobScoreData[].
    scores: JobScoreData[],
    pathwayData: PathwayData,
}

/**
 * ReactElement wrapping the process of picking disease pathway or protein scores to select.
 */
const ScorePicker = ({scores, pathwayData}: ScorePickerParms) => {
    const scoreEls = scores.map((data, index) => {
        const onRemove = () => {
            scores.splice(index, 1);
        };
        return (<ScoreCard onRemove={onRemove} index={index} {...data} />);
    });

    // This is a bit of a hack that allows us to update multiple scores asynchronously,
    // without losing track of the score object.
    // This won't work if something else also updates it in the middle, though.
    const holder = {scores: scores}
    const onSelected = async ({label, jobId, wsId, parms, scoretype, data, weights, groupLabel, ...rest}) => {
        if (scoretype == 'joblist') {
            for (const subscore of data) {
                onSelected(subscore);
            }
            return;
        }

        weights = weights || [];
        const scoreData = await fetchScores(wsId, jobId);

        const [header, pwdata] = scoreData.data;

        const pwToScore: {[key: string]: number[]} = {}
        let minScore = 1e99;
        let maxScore = -1e99;
        let minQScore = 1e99;
        let maxQScore = -1e99;
        for (const [pw, pwScores] of pwdata) {
            const febeQ = pwScores[2];
            const wFEBE = pwScores[5];
            minScore = Math.min(minScore, wFEBE);
            maxScore = Math.max(maxScore, wFEBE);
            minQScore = Math.min(minQScore, febeQ);
            maxQScore = Math.max(maxQScore, febeQ);
            pwToScore[pw] = pwScores
        }

        const weight = weights.length ? _.mean(weights) : 1;
        (holder.scores as any) = holder.scores.push({
            header,
            pwToScore,
            minScore,
            maxScore,
            minQScore,
            maxQScore,
            title: label,
            wsId: wsId,
            jobId: jobId,
            weights: weights,
            weight: weight,
            groupLabel: groupLabel,
        });
    };

    const onProtScoreSelected = async ({label, jobId, wsId, code, parms, scoretype, data, weights, ...rest}) => {
        console.info("Selected with", label, jobId, wsId, data);
        if (scoretype == 'joblist') {
            for (const subscore of data) {
                onProtScoreSelected(subscore);
            }
            return;
        }

        const scoreData = await fetchScores(wsId, jobId, [code]);

        const [header, protdata] = scoreData.data;

        const protToScore: {[key: string]: number} = {}
        let minScore = 1e99;
        let maxScore = -1e99;
        for (const [prot, scores] of protdata) {
            const score = scores[0];
            minScore = Math.min(minScore, score);
            maxScore = Math.max(maxScore, score);
            protToScore[prot] = score;
        }

        protdata.sort(([protA, scoresA], [protB, scoresB]) => {
            return scoresB[0] - scoresA[0];
        });
        const protToRank: {[key: string]: number} = {}
        for (let i = 0; i < protdata.length; ++i) {
            const [prot, scores] = protdata[i];
            protToRank[prot] = 1.0 - i / protdata.length;
        }
        const weight = (weights && weights.length) ? _.mean(weights) : 1;

        // We don't have types for freezerjs, but it returns the array type instead
        // of a length here; use an any cast to bypass.
        (holder.scores as any) = holder.scores.push({
            header,
            protToScore,
            protToRank,
            minScore,
            maxScore,
            title: label,
            wsId: wsId,
            jobId: jobId,
            norm: 'minmax',
            weights: weights,
            weight: weight,
        });
    };

    function clearScores() {
        holder.scores.splice(0, holder.scores.length);
    }

    return (
        <div className='panel panel-primary'>
            <div className='panel-heading'>
                Scores 
                <HoverInfo>
                    Each added score will display a partially-filled-in color-coded rectangle to the right of each pathway<br/>
                    The filled-in height indicates the score value<br/>
                    The left-most rectangles display the aggregate pathway or protein score for that pathway across all added scores<br/>
                    Adding a 'wf' score will add all N jobs from a refresh workflow and weight them in the aggregation based on wzs weights<br/>
                </HoverInfo>
                <a style={clearStyle} onClick={clearScores}>Clear</a>
            </div>
            <div className='panel-body'>
                <div className="row">
                    <span className='col-sm-2'>
                    <GlfPicker onSelected={onSelected}/>
                    <ProtScorePicker onSelected={onProtScoreSelected}/>
                    </span>
                </div>
                <div className="row">
                    {scoreEls}
                </div>
            </div>
        </div>
    );
};

const treeStyle = {
	tree: {
		base: {
			backgroundColor: '#fff',
			color: '#000',
		},
		node: {
			base: {
				position: 'relative',
			},
			activeLink: {
				backgroundColor: '#e1f0ff',
			},
            toggle: {
                base: {
                    position: 'relative',
                    display: 'inline-block',
                    verticalAlign: 'top', marginLeft: '-5px',
                    height: '24px',
                    width: '24px'
                },
                wrapper: {
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    margin: '-7px 0 0 -7px',
                    height: '14px'
                },
                height: 14,
                width: 14,
                arrow: {
                    fill: '#9DA5AB',
                    strokeWidth: 0
                }
            },
			header: {
				base: {
                    display: 'inline-block',
					color: '#000',
                    verticalAlign: 'top',
				}

			}
		},
	}
};

const rankStyle = {
    fontSize: '75%',
    fontWeight: 'bold',
}


const initialData = {
    name: 'root',
    toggled: false,
    children: [],
    loaded: false,
    id: kRootNodeId,
    type: 'pathway',
}

function pathwayUrl(id) {
    if (_.startsWith(id, 'GO:')) {
        return `https://www.ebi.ac.uk/QuickGO/term/${id}`;
    } else {
        return `https://reactome.org/content/detail/${id}`;
    }
}


/**
 * ReactElement summarizing the currently selected pathway
 * (Displays the ancestors and a link to the external reference for this pathway)
 */
const PathwaySummary = ({node, callbacks}) => {
    if (node.id == kRootNodeId) {
        return null;
    }
    node = node.type == 'prot' ? node.parent : node;
    const link = (<a className='hier-link' target='_blank' href={pathwayUrl(node.id)}>
        <b>{node.name} ({node.id}) <span className="glyphicon glyphicon-new-window" /></b>
                </a>
                 );
    const hier = [];
    let curNode = node.parent;
    while (curNode && curNode.id != kRootNodeId) {
        const nodeId = curNode.id;
        const onClick = () => {
            callbacks.selectPathwayById(nodeId);
        }
        const curLink = <a onClick={onClick} className='hier-link'>{curNode.name}</a>
        hier.push(<span>{curLink} &gt; </span>);
        curNode = curNode.parent;
    }
    hier.reverse();

    return (
        <div className='pathway-summary'>
            {hier}
            {link}
        </div>
    );
};

/**
 * ReactElement
 * Displays the reactome visual for this pathway, if available.
 */
const PathwayDiagram = ({node}) => {
    let displayNode = node;
    while (displayNode && !displayNode.hasDiagram) {
        displayNode = displayNode.parent;
    }
    if (displayNode) {
        return (
            <div style={{width: '100%'}}>
                <img style={{width: '100%'}} src={`/publish/pw/${displayNode.id}.svg`}/>
            </div>
        );
    } else {
        return <img />;
    }
};

/**
 * Given an ID-to-rank mapping, this assigns the rank to each appropriate node in the hierarchy, and determines
 * whether each node should be visible given the current settings.
 */
function assignRanksAndVis({idToRank, node, hierarchy, filterer, parentFilteredOut, computeOnly, cache}) {
    // With the new Gene Ontology data, there are a lot of repeated nodes in the hierarchy, which
    // dominate the computation time trying to naively traverse the hierarchy.
    // To prevent that, once we've reached the compute-only part of the tree, we use this to cache
    // results so that we don't need to retraverse for each path through a node.
    cache = cache || {};

    const rank = idToRank[node.id];
    let bestSubtreeNode = {
        rank: rank,
        id: node.id,
    };

    let visibleChild = false;
    let children = null;
    if (node.children !== undefined) {
        if (node.loaded) {
            children = node.children;
        } else {
            // Currently the node graph is built incrementally (which is probably not worth it).
            // We still need to know all the children to know if we should render, so to work around it, we will
            // create fake children with appropriate properties for the algorithm to determine if any are visible.
            // (NOTE: Even though we no longer fetch the graph incrementally, it's still not fully constructed)
            children = _.map(hierarchy[node.id], (childId: string) => ({id: childId, children: [], loaded: false, pseudo: true}));
        }
    }
    if (node.pseudo) {
        computeOnly = true;
    }
    if (computeOnly && node.id in cache) {
        return cache[node.id];
    }
    const selfFilteredOut = !filterer(node.id);
    if (children) {
        for (const child of children) {
            const [childVis, childSubRank] = assignRanksAndVis({
                node: child,
                parentFilteredOut: selfFilteredOut,
                idToRank,
                hierarchy,
                filterer,
                computeOnly,
                cache,
            });
            visibleChild |= childVis;
            bestSubtreeNode = _.minBy([bestSubtreeNode, childSubRank], (x) => x.rank);
        }
    }

    const visible = node.type == 'prot' || visibleChild || !selfFilteredOut;

    if (!computeOnly) {
        node.visible = visible;
        if (node.type == 'prot') {
            node.visible = !parentFilteredOut;
            node.fade = false;
        } else {
            if (selfFilteredOut) {
                node.fade = true;
            } else {
                node.fade = false;
            }
        }

        node.rank = rank;
        node.bestSubtreeNode = bestSubtreeNode;
        node.bestSubtreeRank = bestSubtreeNode.rank;
    }

    cache[node.id] = [visible, bestSubtreeNode];
    return cache[node.id];
}

/**
 * Computes a rank for each pathway, and outputs it as a map {id: rank}
 */
function computeRanks({scores, pathScorer, allPathways) {
    const idToScore = {};
    for (const pw of allPathways) {
        idToScore[pw] = pathScorer(pw);
    }

    const aggScores = Object.entries(idToScore);
    aggScores.sort(([idA, scoreA], [idB, scoreB]) => scoreA - scoreB);

    const out = {}
    let prevScore = undefined;
    let prevRank = 0;
    _.each(aggScores, ([id, score], idx) => {
        let rank;
        if (score == prevScore) {
            rank = prevRank;
        } else {
            rank = idx + 1;
            prevRank = rank;
            prevScore = score;
        }
        out[id] = rank;
    });

    return out;
}

/**
 * Returns a function that can be used to compute the score for a node given its id, based on
 * the currently configured scoring settings.
 * 
 * Parameter sortBy will override config.sortBy
 */
function makeNodeScorer({scorers, config, storeData, sortBy}) {
    const {pathScorer, protScorer, protAgg} = scorers;

    const molScore = (id) => {
        let sum = 0;
        for (const collection of storeData.collections) {
            if (id in collection.pathways) {
                sum += 1;
            }
        }
        return sum;
    };
    const pwyScore = (id) => {
        let sum = 0;
        for (const score of storeData.scores) {
            if (!score.pwToScore) {
                // Prot based, ignore for now...
                continue;
            }
            const weight = score.weight;
            sum += weight * pathScorer(id, score);
        }
        return sum;
    };
    const protScore = (id) => {
        let sum = 0;
        for (const score of storeData.scores) {
            if (!score.pwToScore) {
                const weight = score.weight;
                sum += weight * protAgg(protScorer(id, score, config), true);
            }
        }
        return sum;
    };

    let keyFn;
    const sort = sortBy || config.sortBy;
    if (sort == "molecule") {
        keyFn = (id) => -molScore(id);
    } else if (sort == 'pwyscore') {
        keyFn = (id) => -pwyScore(id);
    } else if (sort == 'protscore') {
        keyFn = (id) => -protScore(id);
    } else if (sort == 'molpwyscore') {
        keyFn = (id) => -(molScore(id) * pwyScore(id));
    } else if (sort == 'molprotscore') {
        keyFn = (id) => -(molScore(id) * protScore(id));
    } else if (sort == 'name') {
        // The secondary sort is already by name, don't reorder.
        keyFn = (x) => 0;
    } else {
        console.error("No sort specified", sort);
    }

    return keyFn;
}

/**
 * Sorts a list of nodes
 */
function sortNodes({nodes, subtreeSort}) {
    // Sort is stable, do alphabetical first.
    nodes = _.sortBy(nodes, 'name');
    const rankProp = subtreeSort == 'self' ? 'rank' : 'bestSubtreeRank';
    nodes = _.sortBy(nodes, (node) => node[rankProp]);
    return nodes;
}

/**
 * Sorts node's list of children, and their children, and so on.
 */
function sortAll({node, subtreeSort}) {
    if (!node || !node.children) {
        return;
    }

    node.children = sortNodes({nodes: node.children, subtreeSort});
    for (const child of node.children) {
        sortAll({node:child, subtreeSort});
    }
}


/**
 * Returns a function that determines whether a pathway/node ID should be visible, given current settings.
 */
function makePathwayFilterer({scores, scorers, storeData, config, allPathways, filters, extraFilterFuncs}) {
    extraFilterFuncs = extraFilterFuncs || [];
    const filterFuncs = [...extraFilterFuncs];
    for (const filter of filters) {
        const thresh = parseFloat(filter.value);
        if (isNaN(thresh)) {
            continue;
        }
        const nodeScorer = makeNodeScorer({sortBy: filter.filterBy, scorers, storeData, config});
        const pathwayToRank = computeRanks({
            scores: scores,
            pathScorer: nodeScorer,
            allPathways,
        });
        const doFilter = (id) => {
            return pathwayToRank[id] <= thresh;
        };
        filterFuncs.push(doFilter);
        
    }
    return (id) => {
        for (const filterFunc of filterFuncs) {
            if (!filterFunc(id)) {
                return false;
            }
        }
        return true;
    };
}

export interface PathwayConfig {
    protScoreNorm: string,
    sortBy: string,
    subtreeSort: string,
    protAgg: string,
    pathwayScoreType: string,
    dedupeType: string,
    febeQFilter: number,
}

interface PathwayTreeParms {
    topStore: any,
    pathwayData: PathwayData,
}

/**
 * Somewhat misleadingly named, this is the main component displaying everything.
 * This includes the pathway network on the left (via Treebeard), and the details views
 * on the right (which are mostly delegated to other elements.
 */
const PathwayTree = ({topStore, pathwayData}: PathwayTreeParms) => {
    console.info("Start redraw");
    const [data, setData] = useState(initialData);
    let [cursor, setCursor] = useState<PathwayNode|boolean>(false);
    const [config, setConfig] = useState<PathwayConfig>({
        protScoreNorm: 'minmax',
        sortBy: 'molecule',
        subtreeSort: 'self',
        protAgg: 'scoreplot',
        pathwayScoreType: 'default',
        dedupeType: 'none',
        febeQFilter: -2,
    });
    const scorers = Score.makeScorers(config, pathwayData);

    window.pathwayData = pathwayData;


    const fetchNode = async (data, node: PathwayNode, skipSet) => {
        console.info("Fetch ", node.id, node.name, node.type);
        node.loading = true;

        let childNodes = [];
        const pwChildIds = pathwayData.hierarchy[node.id];
        if (node.type != 'prot') {
            if (pwChildIds) {
                childNodes = pwChildIds.map((id: string) => pathwayData.pathways[id]);
                childNodes = childNodes.filter((child) => pathwayData.protsets[child.id] !== undefined);
            } else {
                const protSet = pathwayData.protsets[node.id];
                for (const prot of protSet) {
                    childNodes.push({
                        id: prot,
                        name: `${pathwayData.prot2gene[prot]} (${prot})`,
                        type: 'prot',
                    });
                }
            }
        }

        const children = []
        for (const childNode of childNodes) {
            const {id, name, type, hasDiagram, ...rest} = childNode;
            if (id == node.id) {
                continue;
            }
            children.push({
                id,
                name,
                type,
                hasDiagram,
                toggled: false,
                children: type == 'prot' ? undefined : [],
                loaded: false,
                parent: node,
            });
        }
        node.children = children;
        node.loading = false;
        node.loaded = true;

        if (!skipSet) {
            console.info("Set data in fetchNode");
            setData({...data});
        }
    };

    const onToggle = async (node: PathwayNode, toggled: boolean, skipSet) => {
        if (cursor) {
            cursor.active = false;
        }
        node.active = true;
        if (node.children) {
            node.toggled = toggled;
        }
        if (!skipSet) {
            console.info("Set data in onToggle");
            setCursor(node);
            setData({...data});
        } else {
            cursor = node;
        }
        if (!node.loaded && !node.loading) {
            await fetchNode(data, node, skipSet);
        }
        console.info("Toggled", node, toggled, node.toggled);
    };

    if (data['name'] == 'root' && !data.toggled && !data.loading && !data.loaded) {
        onToggle(data, true, false);
    }

    const store = topStore.get();

    let [storeData, setStoreData] = useState(topStore.get());

    const collapseAll = (node) => {
        // Don't collapse root, despite the name, leave it at level - 1.
        if (node['name'] != 'root') {
            node.toggled = false;
        }

        if (node.children) {
            for (const child of node.children) {
                collapseAll(child);
            }
        }

        if (node['name'] == 'root') {
            setData({...data});
        }
    };

    const openAll = async ({subtree, idMap, skipSet, foundNodes}) => {
        // Opens all nodes in the pathway that are listed in idMap.
        // Will only load visible loads (i.e. if there's a leaf in idmap,
        // make sure all its parents are in there too).
        if (subtree.id != kRootNodeId && !(subtree.id in idMap)) {
            return foundNodes;
        }

        foundNodes[subtree.id] = subtree;

        if (!subtree.toggled) {
            console.info("Toggling open", subtree);
            await onToggle(subtree, true, true);
        }
        
        console.info("Opening all from", subtree.id);
        if (subtree.children) {
            for (const child of subtree.children) {
                await openAll({subtree: child, skipSet:true, idMap, foundNodes});
            }
        }

        if (!skipSet) {
            console.info("Setting final data");
            setData({...data});
        }
        return foundNodes;
    };

    const selectPathwayById = async (id) => {
        console.info("Selecting pathway", id);
        const idMap = {};
        findPathwayAncestry({ idToFind: id, hierarchy: pathwayData.hierarchy, out: idMap});
        console.info("Pathway ancestry contains:", idMap);
        const foundNodes = await openAll({subtree:data, foundNodes:{}, skipSet: true, idMap});
        if (id in foundNodes) {
            console.info("Highlight selected pathway");
            cursor.active = false;
            const node = foundNodes[id];
            node.active = true;
            // Supposedly future reacts might batch automatically here.
            ReactDOM.unstable_batchedUpdates(() => {
                setCursor(node);
                setData({...data});

                setTimeout(() => {
                    document.getElementById('active-node').scrollIntoView({
                        behavior: 'smooth',
                        block: 'nearest',
                    });
                }, 0);
            });
        }
    };



    useEffect(() => {
        topStore.on('update', () => {
            console.info("Updated store data");
            setStoreData(topStore.get());

            setData({...data});

            saveSession(topStore.get());
        });

        return () => {
            topStore.off('update');
        };
    });

    const [initialPathway, setInitialPathway] = useState(storeData['initPathway']);
    const [callbacks, setCallbacks] = useState({});
    const [filters, setFilters] = useState([]);

    if (initialPathway) {
        setInitialPathway(null);
        console.info("Selecting initial pathway", initialPathway);
        selectPathwayById(initialPathway);
    }

    // We disable the default toggle and render it ourselves in Header, to have more control over it.
    const Toggle = ({node, ...props}) => {
        return null;
    };

    const Container = ({node, ...props}) => {
        if (node.visible) {
            return (<decorators.Container node={node} {...props} animations={false} />);
        } else {
            return (<span></span>);
        }
    };

    const Header = ({onSelect, node, style, customStyles}) => {
        const prefix = [];
        if (node.type != 'prot') {
            if (node.toggled) {
                prefix.push(<span className="glyphicon glyphicon-chevron-down tree-toggle" />);
            } else {
                prefix.push(<span className="glyphicon glyphicon-chevron-right tree-toggle" />);
            }
        } else {
            // Add in a spacer
            prefix.push(<span style={{'margin-right': '1rem'}} />);
        }
        let idx = 0;
        for (const collection of storeData.collections) {
            const color = kColorList[idx % kColorList.length];
            prefix.push(<CollectionMarker node={node} collection={collection} color={color} />);
            idx += 1;
        }

        let tail = [];

        tail.push(<AggScoreMarker node={node} scores={storeData.scores} scorers={scorers} />);

        idx = 0;
        const norm = config.protScoreNorm;
        for (const score of storeData.scores) {
            const color = kColorList[idx % kColorList.length];
            tail.push(<ScoreMarker
                node={node}
                score={score}
                color={color}
                scorers={scorers}
                />);
            idx += 1;
        }

        let name = node.name;
        const kMaxLength = 40;
        if (name.length > kMaxLength) {
            name = name.substring(0, kMaxLength - 3) + '...';
        }

        const curStyle = {...style.base};
        if (node.fade) {
            curStyle.color = '#999';
        }
        const extraProps = {};
        if (node.active && node.id != kRootNodeId) {
            extraProps['id'] = 'active-node';
        }
        const subtreeClick = (e) => {
            console.info("Going to best subtree", node.bestSubtreeNode);
            selectPathwayById(node.bestSubtreeNode.id);
            // This will default to 'clicking' the current pathway, which can toggle and prevent showing.
            e.stopPropagation();
        }
        const subtreeLink = (<a onClick={subtreeClick}>{node.bestSubtreeRank}</a>);

        return (<div style={curStyle} onClick={onSelect} {...extraProps}>
            <div style={node.selected ? {...style.title, ...customStyles.header.title} : style.title}>
                {prefix}&nbsp;
                <span className='hoverable' style={rankStyle}>
                    {node.rank}
                    <span className='hover-text'>
                        Rank: {node.rank}<br/>
                        Subtree Rank: {subtreeLink}</span>
                </span>&nbsp;
                {name}&nbsp;
                {tail}
            </div>
        </div>);
    };

    const decorator = {
        ...decorators,
        Header: Header,
        Container: Container,
        Toggle: Toggle,
    };

    const diaStyle = {
        height: '100%',
        borderLeft: '1px solid #eee',
        position: 'sticky',
        top: 0,
    };
    const fullStyle = {
        marginTop: '1rem',
	    boxShadow: '2px 2px 8px 1px rgba(100, 100, 100, 0.3)',
    };

    const onCfg = (name: string, e) => {
        const newCfg = {...config};
        newCfg[name] = e.target.value;
        setConfig(newCfg);
    };
    const onSortBy = onCfg.bind(this, 'sortBy');
    const onSubtreeSortBy = onCfg.bind(this, 'subtreeSort');
    const onProtAgg = onCfg.bind(this, 'protAgg');
    const onProtScoreNorm = onCfg.bind(this, 'protScoreNorm');
    const onPathwayScoreType = onCfg.bind(this, 'pathwayScoreType');
    const onDedupeType = onCfg.bind(this, 'dedupeType');
    const onFebeQFilter = onCfg.bind(this, 'febeQFilter');


    const nodeScorer = makeNodeScorer({scorers, storeData, config});
    const allPathways = Object.keys(pathwayData.protsets);
    const pathwayToRank = computeRanks({
        scores: storeData.scores,
        pathScorer: nodeScorer,
        allPathways,
    });

    const extraFilterFuncs = [];
    if (config.dedupeType != 'none') {
        const dedupeMap = makeDedupeMap({
            protsets:pathwayData.protsets,
            hierarchy:pathwayData.hierarchy,
            id:kRootNodeId,
            siblingsOnly: config.dedupeType == 'dedupe-siblings',
        });
        extraFilterFuncs.push((id) => {
            return !(id in dedupeMap && dedupeMap[id] != id);
        });
    }
    const filterer = makePathwayFilterer({scores: storeData.scores, scorers, storeData, config, allPathways, filters, extraFilterFuncs});

    console.info("Start assign");
    assignRanksAndVis({node: data, idToRank: pathwayToRank, filterer: filterer, hierarchy: pathwayData.hierarchy});
    console.info("End assign");
    sortAll({
        node:data,
        subtreeSort:config.subtreeSort,
     });

    const rowStyle = {
        display: 'flex',
    };
    const configItemStyle = {
        marginRight: '1rem',
    }

    console.info("Tree redraw");

    callbacks.selectPathwayById = selectPathwayById;

    const [tab, setTab] = useState('diagram');
    let rightPanel;
    let diaBtnStyle = 'btn btn-default';
    let heatBtnStyle = 'btn btn-default';
    let scatterBtnStyle = 'btn btn-default';
    let scoreDetailsBtnStyle = 'btn btn-default';
    let scoreCompareBtnStyle = 'btn btn-default';
    if (tab == 'diagram') {
        diaBtnStyle='btn btn-info';
        rightPanel = <PathwayDiagram node={cursor} />;
    } else if (tab == 'heatmap') {
        heatBtnStyle='btn btn-info';
        rightPanel = <PathwayHeatmap storeData={storeData} scorers={scorers} pathwayData={pathwayData} plotType="heatmap" callbacks={callbacks} />;
    } else if (tab == 'scatter') {
        scatterBtnStyle='btn btn-info';
        rightPanel = <PathwayHeatmap storeData={storeData} scorers={scorers} pathwayData={pathwayData} plotType="scatter" callbacks={callbacks} cursorPathwayId={cursor.id} />;
    } else if (tab == 'scoreDetails') {
        scoreDetailsBtnStyle='btn btn-info';
        rightPanel = <ScoreDetails storeData={storeData} pathwayData={pathwayData} pathwayId={cursor.id} scorers={scorers} config={config} callbacks={callbacks} pathwayFilterer={filterer}/>;
    } else if (tab == 'scoreCompare') {
        scoreCompareBtnStyle='btn btn-info';
        rightPanel = <PathwayCompare storeData={storeData} scorers={scorers} pathwayData={pathwayData} callbacks={callbacks} cursorPathwayId={cursor.id} />;
    }

    console.info("Done redraw");

    return (
        <div className='container-fluid'>
            <TargetPicker pathwaysData={pathwayData} collections={storeData.collections}/>
            <ScorePicker scores={storeData.scores} pathwayData={pathwayData}/>
            <div className='panel panel-primary'>
                <div className='panel-heading'>Config</div>
                <div className='panel-body'>
                    <div style={rowStyle}>
                        <div style={configItemStyle}>
                            
                            Sort:
                            <HoverInfo>
                                Determines both the ordering and the displayed rank for each pathway, based on:<br/>
                                <b># Molecules</b>: Number of molecules (or target groups) that directly hit it<br/>
                                <b>Pathway Score</b>: Aggregate PathwayScoreType selected (usually GLF) across all pathway scores<br/>
                                <b>Protein Score</b>: Aggregate pathway protein score, based on selected Prot Norm and ProtAgg<br/>
                                <b>#Mol * X Score</b>: Multiplies two of the above scores together<br/>
                            </HoverInfo>
                            <select style={{width: 'auto'}} className='form-control' value={config.sortBy} onChange={onSortBy}>
                                <option value='molecule'># Molecules</option>
                                <option value='pwyscore'>Pathway Score</option>
                                <option value='protscore'>Protein Score</option>
                                <option value='molpwyscore'>#Mol*Pwy Score</option>
                                <option value='molprotscore'>#Mol*Prot Score</option>
                                <option value='name'>Name</option>
                            </select>
                        </div>
                        <div style={configItemStyle}>
                            SubtreeSort: 
                            <HoverInfo>
                                Non-leaf pathways are ordered by <br/>
                                <b>Self</b>: their own score<br/>
                                <b>Best Descendent</b>: the best score of any of their nested pathways<br/>
                            </HoverInfo>
                            <select style={{width: 'auto'}} className='form-control' value={config.subtreeSort} onChange={onSubtreeSortBy}>
                                <option value='self'>Self</option>
                                <option value='best_descendant'>Best Descendant</option>
                            </select>
                        </div>
                        <div style={configItemStyle}>
                            Prot Norm:
                            <HoverInfo>
                                Proteins within a pathway are scored as<br/>
                                <b>MinMax</b>: 0-1 normalized separately for each protein score<br/>
                                <b>Rank</b>: Rank of the protein within each protein score<br/>
                            </HoverInfo>
                            <select style={{width: 'auto'}} className='form-control' value={config.protScoreNorm} onChange={onProtScoreNorm}>
                                <option value='minmax'>MinMax</option>
                                <option value='rank'>Rank</option>
                            </select>
                        </div>
                        <div style={configItemStyle}>

                            Prot Agg:
                            <HoverInfo>
                                How to compute the pathway aggregate protein score, for each separate protein score.<br/>
                                ScorePlot uses mean for sorting purposes, but displays a mini scoreplot in the UI.
                            </HoverInfo>
                            <select style={{width: 'auto'}} className='form-control' value={config.protAgg} onChange={onProtAgg}>
                                <option value='scoreplot'>ScorePlot</option>
                                <option value='mean'>Mean</option>
                                <option value='median'>Median</option>
                                <option value='max'>Max</option>
                            </select>
                        </div>
                        <div style={configItemStyle}>
                            PathwayScoreType:
                            <HoverInfo>
                                Which pathway score value to use and display<br/>
                                Default is a wFEBE odds-ratio based score.
                            </HoverInfo>
                            <select style={{width: 'auto'}} className='form-control' value={config.pathwayScoreType} onChange={onPathwayScoreType}>
                                <option value='default'>Default</option>
                                <option value='febeQ'>FebeQ (log10)</option>
                            </select>
                        </div>
                        <div style={configItemStyle}>
                            Dedupe:
                            <HoverInfo>
                                Hide pathways with identical protein sets.<br/>
                                Siblings will only consider adjacent nodes for dedupe.<br/>
                                All will preserve only 1 (arbitrary) instance of each protein set across the entire tree.
                            </HoverInfo>
                            <select style={{width: 'auto'}} className='form-control' value={config.dedupeType} onChange={onDedupeType}>
                                <option value='none'>None</option>
                                <option value='dedupe-siblings'>Siblings</option>
                                <option value='dedupe-all'>All</option>
                            </select>
                        </div>
                        <div style={configItemStyle}>
                            FebeQ (-log10):
                            <HoverInfo>
                                Filters (sets to 0) pathway scores whose Q-value is greater than this.<br/>
                                We score all pathways, but typically in code only make use of those scores that pass a Q threshold.
                            </HoverInfo>
                            <input className='form-control' type='number' value={config.febeQFilter} onChange={onFebeQFilter} />
                        </div>
                        <PathwayFilters filters={filters} setFilters={setFilters} />
                            <HoverInfo>
                                Filters limit which pathways display in the tree<br/>
                                Multiple filters and "AND"d together.
                            </HoverInfo>
                    </div>
                </div>
            </div>
            <div className='row' style={fullStyle}>
                <div className='col-lg-4' style={{'height': '95vh', 'overflow': 'auto'}}>
                    <div><a style={{'cursor': 'pointer'}} onClick={() => collapseAll(data)}>Collapse All</a></div>
                    {/* Animations are disabled because otherwise scrollIntoView is very hard. */}
                    <Treebeard
                        style={treeStyle}
                        data={data}
                        onToggle={onToggle}
                        decorators={decorator}
                        animations={false}
                    />
                </div>
                <div className='col-lg-8' style={diaStyle}>
                    <PathwaySummary node={cursor} callbacks={callbacks} />
                    <div className='btn-group'>
                    <button onClick={()=>setTab('diagram')}  className={diaBtnStyle}>Diagram</button>
                    <button onClick={()=>setTab('heatmap')}  className={heatBtnStyle}>Heatmap</button>
                    <button onClick={()=>setTab('scatter')}  className={scatterBtnStyle}>Scatter</button>
                    <button onClick={()=>setTab('scoreDetails')}  className={scoreDetailsBtnStyle}>Score Details</button>
                    <button onClick={()=>setTab('scoreCompare')}  className={scoreCompareBtnStyle}>Compare</button>
                    </div>
                    <hr/>
                    {rightPanel}
                </div>
            </div>
        </div>
    );
}


/**
 * Initializes the page, in particular selecting anything that is supposed to be selected based on
 * URL parameters (e.g. if the user got linked here to look at a specific pathway, or with a drug preloaded.)
 */
function initSetup(topStore: any, pathwaysData: PathwayData) {
    const collections = topStore.get().collections;
    // Note that this is async, we don't wait for it before rendering.
    const init = loadInitFromHash();
    if (init['initWsa']) {
        console.info("Loading initial WSA");
        async function loadWsaAsync() {
            const resp = await fetch(`/api/wsa/${init.initWsa}/`);
            const respData = await resp.json();
            const mol = respData.data;
            const targets = mol.targets.map((x) => {
                return {
                    uniprot: x[0],
                    gene: x[1],
                    direction: x[2],
                };
            });
            addTargetCollection(pathwaysData, collections, mol.name, targets);
        }
        loadWsaAsync();
    }
    if (init['initProt'] && init['initGene']) {
        const targets = [{
            uniprot: init.initProt,
            gene: init.initGene,
            direction: 0,
        }];
        addTargetCollection(pathwaysData, collections, init.initGene, targets);
    }

    if (init['initPathway']) {
        console.info("Loading initial pathway");
        topStore.get().set('initPathway', init['initPathway']);

    }

    if (Object.keys(init).length  == 0) {
        console.info("Reloading from session");
        const session = loadSession();
        if (session) {
            topStore.get().set(session);
        }
    }
}


/**
 * Toplevel external entrypoint for putting the PathwayTree into the scene.
 */
export function drawPathwayTree(el: HTMLElement) {
    const topStore = new Freezer({
        collections: [],
        scores: [],
        initPathway: null,
    });

    ReactDOM.render(<div>Loading...</div>, el);

    fetchPathwayData().then((pathwayData: PathwayData) => {
        initSetup(topStore, pathwayData);
        ReactDOM.render(<PathwayTree topStore={topStore} pathwayData={pathwayData}/>, el);
    });
}
