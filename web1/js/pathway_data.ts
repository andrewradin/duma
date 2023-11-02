

// We use settings to encourage pathway data caching locally because it's a fairly
// large/slow load.
// If the content changes, this version should be bumped to force a cache miss.
const PW_DATA_VER = 1;

export const kRootNodeId = '[root]';

export type ProtSet = string[];
export type Hierarchy = {[key: string]: string[]};

export interface PathwayNode {
    id: string;
    type: string;
    name: string;
    hasDiagram: boolean;
    active?: boolean;
    loading?: boolean;
    loaded?: boolean;
    children?: PathwayNode[];
}

export interface PathwayData {
    protsets: {[key: string]: ProtSet},
    idToName: {[key: string]: string},
    hierarchy: Hierarchy,
    pathways: {[key: string]: PathwayNode},
    prot2gene: {[key: string]: string},
    protToPathways: {[key: string]: string[]}
}

export async function fetchPathwayData() {
    const resp = await fetch(`/api/pathway_data/?v=${PW_DATA_VER}`, {'cache': 'force-cache'});
    const respData = await resp.json();
    const pathwayData: PathwayData = respData.data;
    processPathwayData(pathwayData);
    return pathwayData;
}

/**
 * Some data we don't send down and instead preprocess.
 * (We send down pathway->prot, and precompute prot->pathway)
 */
function processPathwayData(pathwayData: PathwayData) {
    const protToPathways: {[key: string]: string[]} = {};
    for (const [pathway, uniprots] of Object.entries(pathwayData.protsets)) {
        for (const uniprot of uniprots) {
            if (!(uniprot in protToPathways)) {
                protToPathways[uniprot] = [];
            }
            protToPathways[uniprot].push(pathway);
        }
    }

    pathwayData.protToPathways = protToPathways;
}