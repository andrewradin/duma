

const GE = 'Gene Expression';
const GWAS = 'GWAS';
const otarg = 'Open Targets';
const dgn = 'DisGeNet';
const clin = 'Clinical';
const tcgamut = 'TCGA Mutation';
const meta = 'meta';
const sig = 'sig';
const empty = '';

const faers = 'FAERS';
const cc = 'Case/Control GE';
const mirna = 'microRNA';

const path = 'PathSum';
const gesig = 'GE Signature'
const direct = 'direct';
const indirect = 'indirect';
const direction = 'direction';
const gpbr = 'Background Removal';
const codes = 'CoDES';
const glf = 'GLF';
const sigdif = 'Signature Diffuser';
const esga = 'ESGA';
const gpath = 'gPath';
const gwasig = 'GWAS Signature';
const depend = 'DEPEND';
const capp = 'Comorbidity Pathsum';
const defus = 'DEFUS';
const faerssig = 'CAPP Sig';
const dirJac = 'Direct Jaccard';
const indJac = 'Indirect Jaccard';
const prsim = 'PRSim';
const pwOverlap = 'Pathway Overlap';
const indigo = 'Indigo';
const rdkit = 'RDKit';
const agr = 'AGR';
const misig = 'MI Signature';
const mips = 'MI Pathsum';


const start = [mirna, cc, GWAS, faers, otarg, dgn, tcgamut, agr, mips, misig];
const adj = {};
const positions = {};

const CELL_W = 400;
const CELL_H = 25;


function fork(nodes_in, nodes_out) {
    if (Array.isArray(nodes_in)) { 
        for (const node_in of nodes_in) {
            fork(node_in, nodes_out);
        }
        return;
    }
    const node_in = nodes_in;

    if (Array.isArray(nodes_out)) { 
        for (const node_out of nodes_out) {
            fork(node_in, node_out);
        }
        return;
    }
    const node_out = nodes_out;
    
    if (!adj[node_in]) {
        adj[node_in] = [];
    }
    adj[node_in].push(node_out);
}

const termId = 'effagg';

const graphNodes = [{
    data: {
        id: termId,
        label: 'Efficacy Agg'
    }
}]
const graphEdges = []

const terms = [];

let curRow = 0;
function addTerminal(graphNode, origin) {
    curRow += 1;
    const curTermId = 'term' + graphNode;
    const term = {
        data: {
            id: curTermId,
            label: 'Eff Agg Entry',
            origin: origin,
            //parent: termId,
        } 
    };
    positions[curTermId] = {
        // x will be filled after.
        y: positions[graphNode].y,
    }
    terms.push(curTermId);
    graphNodes.push(term);

    graphEdges.push({
        data: {
            id: 'term-e-' + graphNode,
            source: graphNode,
            target: curTermId, 
            etype: 'mol',
        }
    });
    graphEdges.push({
        data: {
            id: 'term-ee-' + graphNode,
            source: curTermId,
            target: termId, 
            etype: 'mol',
        }
    });
}

let maxDepth = 0;

const molTypes = [dirJac, indJac, prsim, indigo, rdkit, defus, gpbr];
const otherTypes = [meta, clin, sig, capp];

const scoreInType = {};
for (const type of molTypes) {
    scoreInType[type] = 'mol';
};
for (const type of otherTypes) {
    scoreInType[type] = 'other';
};
scoreInType[depend] = 'pwy';


const multiType = [capp, mirna, cc, meta, sig, GWAS, capp];

// Manual depth assignment is a bit fragile.
// If the diagram changes, you might want to disable this and stare at the
// graph to come up with new assignments.
const forceDepth = {};
forceDepth[depend] = 5;
forceDepth[codes] = 4;
forceDepth[gpbr] = 5;
forceDepth[glf] = 4;
forceDepth[sigdif] = 3;

function addGraphNode(label, prevGraphNode, origin, depth) {
    if (forceDepth[label]) {
        depth = forceDepth[label];
    }
    maxDepth = Math.max(maxDepth, depth);
    const fullId = label + graphNodes.length;

    const data = {
            id: fullId,
            label: label,
            origin: origin,
        } ;
    
    if (multiType.indexOf(label) != -1) {
        data.effect = 'ghost';
    } else {
        data.effect = 'single';
    }

    graphNodes.push({ data });
    positions[fullId] = {
        x: depth * CELL_W,
        y: curRow * CELL_H
    }
    if (prevGraphNode) {
        graphEdges.push({
            data: {
                id: fullId + prevGraphNode,
                source: prevGraphNode,
                target: fullId,
                etype: scoreInType[label] ? scoreInType[label] : 'protein'
            }
        });
    }
    return fullId;
}

const toAttach = {
    'DPI': [direct, indirect, codes, dirJac, direction, prsim, esga],
    'PPI': [indirect, indJac, sigdif, prsim, direction, esga],
    'D2PS': [depend],
    'STRUCT': [indigo, rdkit],
    'RCTOME': [glf, pwOverlap],
    'GO': [glf, pwOverlap],
    'UKBB': [GWAS],
    'GWASCAT': [GWAS],
    'PHEWAS': [GWAS],
    'GRASP': [GWAS],
    'DisGeNet': [capp],
    'OTarg': [capp],
    'GEO': [cc, mirna],
    'ArrayExpress': [cc, mirna],
    'TargetScan': [mirna],
    'Monarch Initiative': [mips, misig],
}
const toAttachRev = {};
for (const [info, nodes] of Object.entries(toAttach)) {
    for (const node of nodes) {
        if (!toAttachRev[node]) {
            toAttachRev[node] = [];
        }
        toAttachRev[node].push(info);
    }
}

const attachments = {}

function addGraphAttachment(label, parentNode) {
    const fullId = label + graphNodes.length;

    graphNodes.push({
        data: {
            id: fullId,
            label: label,
            type: 'miniinput',
        } 
    });

    graphEdges.push({
        data: {
            id: 'e' + fullId + parentNode,
            source: fullId,
            target: parentNode, 
            etype: 'attachment',
        }
    });
    if (!attachments[parentNode]) {
        attachments[parentNode] = []
    }
    attachments[parentNode].push(fullId);
    return fullId;
}

const terminals = [];
function process(nodes, depth, prevGraphNode, origin) {
    if (Array.isArray(nodes)) {
        for (const node of nodes) {
            process(node, depth, prevGraphNode, origin);
        }
        return;
    }
    const node = nodes;
    if (!origin) {
        origin = node;
    }

    if (node == empty) {
        addTerminal(prevGraphNode, origin);
        return;
    }

    const startRow = curRow;
    const graphNode = addGraphNode(node, prevGraphNode, origin, depth);
    
    if (toAttachRev[node]) {
        for (const attach of toAttachRev[node]) {
            addGraphAttachment(attach, graphNode);
        }
    }
    


    if (adj[node]) {
        const nextNodes = [];
        for (const nextNode of adj[node]) {
            nextNodes.push(nextNode);
        }
        process(nextNodes, depth+1, graphNode, origin);
    } else {
        addTerminal(graphNode, origin);
    }

    const endRow = curRow - 1;
    if (endRow - startRow > 1) {
        const midRow = (startRow + endRow - 1) / 2;
        positions[graphNode].y = midRow * CELL_H;
    }
}


const serial = fork;

fork([cc, mirna], meta);
serial(faers, clin);
serial(GE, meta);
serial(meta, sig);
fork(sig, [path, gesig]);
fork(path, [direct, indirect, direction]);
fork([direct, indirect, direction], [empty, gpbr]);
fork(gesig, [codes, glf]);

fork(GWAS, [gpath, esga, gwasig]);
fork(gpath, [direct, indirect]);
fork(gwasig, [sigdif]);

fork(mips, [direct, indirect]);

const subotarg = ['literature', 'rna_expression', 'somatic_mutation', 'genetic_association', 'known_drug', 'animal_model', 'affected_pathway'];

fork(otarg, subotarg);
fork(subotarg.concat([dgn, tcgamut, agr, misig]), [codes, glf, sigdif]);
fork(sigdif, [codes, glf])
serial(glf, depend);

fork(clin, [capp, defus]);
fork(capp, [faerssig, direct, indirect]);
fork(defus, [indJac, prsim, indigo, rdkit, pwOverlap]);

fork(faerssig, [codes, glf]);


process(start, 0);

const effHeight = curRow * CELL_H;

positions['effagg'] = {
    x: (maxDepth + 2) * CELL_W,
    y: effHeight / 2,
}

for (const term of terms) {
    positions[term].x = (maxDepth + 1) * CELL_W;
}

const frontNodes = []

for (const node of graphNodes) {
    if (node.data.effect == 'ghost') {
        for (let i = 1; i <= 3; ++i) {
            const ghostData = Object.assign({}, node.data);
            ghostData.id += `_ghost${i}`;
            ghostData.effect = 'fade';
            ghostData.label = '';

            const anchorPos = positions[node.data.id];

            frontNodes.push({ data: ghostData });
            positions[ghostData.id] = {
                x: anchorPos.x + i * 5,
                y: anchorPos.y + i * 3,
            }
        }
    }
}

// Stick ghosts onto the front in reverse order.
// This is simpler than working with z-index.
frontNodes.reverse();
graphNodes.unshift(...frontNodes);


const NODE_W = 180;
const NODE_H = 20;
const MINI_W = 85
const MINI_H = 12;

for (const [node, children] of Object.entries(attachments)) {
    for (let i = 0; i < children.length; ++i) {
        const child = children[i];
        positions[child] = {
            x: positions[node].x - NODE_W / 2 - MINI_W * 2 / 3,
            y: positions[node].y - NODE_H / 2 + (MINI_H) * (i + 0.25),
        };
    }
}

const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: graphNodes.concat(graphEdges),
    style: [{
            selector: 'node',
            style: {
                shape: 'roundrectangle',
                label: 'data(label)',
                color: '#222',
                'background-color': '#e3e4fa',
                'border-color': 'black',
                'border-width': '1.0px',
                'border-opacity': '1.0',
                'text-valign': 'center',
                'text-halign': 'center',
                width: `${NODE_W}px`,
                height: `${NODE_H}px`,
                'font-weight': 600,
                ghost: 'no',
            },
        }, {
            selector: 'edge',
            style: {
                'target-arrow-color': '#c00',
                'line-color': '#ccc',
                'curve-style': 'bezier',
                'width': '2px',
                'target-arrow-shape': 'triangle-backcurve',
                'line-fill': 'linear-gradient',
                'line-gradient-stop-colors': '#922 #c22',
                'source-endpoint': '90deg',
                'target-endpoint': '270deg',
            }
        }, {
            selector: '[effect="fade"]',
            style: {
                'opacity': 0.5,
                'border-color': '#000',
                'border-width': '1px',
            }
        }, {
            selector: '[etype="mol"]',
            style: {
                'line-gradient-stop-colors': '#229 #22c',
                'target-arrow-color': '#00c',
            }
        }, {
            selector: '[etype="pwy"]',
            style: {
                'line-gradient-stop-colors': '#929 #c2c',
                'target-arrow-color': '#c0c',
            }
        }, {
            selector: '[etype="other"]',
            style: {
                'line-gradient-stop-colors': '#222 #555',
                'target-arrow-color': '#555',
            }
        }, {
            selector: '[id="effagg"]',
            style: {
                height: effHeight / 10,
                'background-color': '#fafafa',
                'border-width': 2,
                shape: 'rectangle',
            }
        }, {
            selector: `[origin="${clin}"],[origin="${faers}"]`,
            style: {
                'background-color': '#e3f4fa',
            }
        }, {
            selector: `[origin="${GE}"],[origin="${mirna}"],[origin="${cc}"]`,
            style: {
                'background-color': '#e3f4ea',
            }
        }, {
            selector: `[label="${GWAS}"]`,
            style: {
            }
        }, {
            selector: `[origin="${GWAS}"]`,
            style: {
                'background-color': '#f3e4fa',
            }
        }, {
            selector: `[type="miniinput"]`,
            style: {
                shape: 'octagon',
                'background-color': '#ffe770',
                'color': '#000',
                width: `${MINI_W}px`,
                height: `${MINI_H}px`,
                'font-size': '12pt',
                'font-weight': 600,
                opacity: 0.6,
            }
        }, {
            selector: '[etype="attachment"]',
            style: {
                'line-color': '#000',
                'curve-style': 'bezier',
                'width': '1px',
                'target-endpoint': '270deg',
                'line-fill': 'solid',
                'target-arrow-color': '#000',
                'opacity': 0.5,
            }
        }, {
            selector: '[label="DPI"]',
            style: {
                'background-color': '#00a',
                'color': '#fff',
            }
        }, {
            selector: '[label="PPI"]',
            style: {
                'background-color': '#a00',
                'color': '#fff',
            }
        }, {
            selector: '[label="D2PS"]',
            style: {
                'background-color': '#a0a',
                'color': '#fff',
            }
        }, {
            selector: '[label="RCTOME"]',
            style: {
                'background-color': '#a0a',
                'color': '#fff',
            }
        }, {
            selector: '[label="STRUCT"]',
            style: {
                'background-color': '#0aa',
                'color': '#fff',
            }
        }

    ],
    layout: {
        name: 'preset',
        positions: positions,
    },
});
