import * as _ from 'lodash';

import {PathwayData} from './pathway_data';
import {PathwayConfig, JobScoreData} from './pathway_tree';

function glfPathScorer(febeQFilter: number, nodeId: string, jobData: JobScoreData): number {
    const score = jobData;
    const maxScore = score.maxScore;
    const minScore = score.minScore;
    febeQFilter = parseFloat(febeQFilter as any);
    if (isNaN(febeQFilter)) {
        febeQFilter = 1e99;
    }
    // The nominal febeq value is expressed in log10; get the float value to use.
    const febeQRealFilter = Math.pow(10, febeQFilter);
    if (nodeId in score.pwToScore!) {
        const scores = score.pwToScore![nodeId];
        const [nProts, setPor, febeQ, febeOR, peakInd, wFEBE] = scores;
        if (febeQ > febeQRealFilter) {
            return 0;
        } else {
            // Offset range to avoid NaNs if minScore == maxScore.
            const h = (wFEBE - minScore) / (maxScore - minScore + 1e-30);
            return h;
        }
    } else {
        return 0;
    }
}

function glfFebeQPathScorer(nodeId: string, jobData: JobScoreData): number {
    const score = jobData;
    // Both signs & min/max are flipped here.
    const maxScore = -Math.log10(score.minQScore!);
    const minScore = -Math.log10(score.maxQScore!);
    if (nodeId in score.pwToScore!) {
        const scores = score.pwToScore![nodeId];
        const [nProts, setPor, febeQ, febeOR, peakInd, wFEBE] = scores;
        const val = -Math.log10(febeQ);
        // Offset range to avoid NaNs if minScore == maxScore.
        const h = (val - minScore) / (maxScore - minScore + 1e-30);
        return h;
    } else {
        return 0;
    }
}

export function makePwScorer(scoreType: string, febeQFilter: number) {
    if (scoreType == 'default') {
        return _.partial(glfPathScorer, febeQFilter);
    } else {
        return glfFebeQPathScorer;
    }
}


export function makeProtScorer(pathwayData: PathwayData, config: PathwayConfig) {
    function protScorer(nodeId: string, jobData: JobScoreData): number[] {
        const score = jobData;
        let maxScore = score.maxScore;
        let minScore = score.minScore;
        const pathwayToProts = pathwayData.protsets;
        const pathProts = pathwayToProts[nodeId];
        const norm = config.protScoreNorm;
        const scoreMap = norm == 'rank' ? score.protToRank : score.protToScore;
        if (norm == 'rank') {
            minScore = 0;
            maxScore = 1;
        }
        if (pathProts) {
            const protscores = [];
            for (const prot of pathProts) {
                if (prot in scoreMap) {
                    protscores.push((scoreMap[prot] - minScore) / (maxScore - minScore));
                } else {
                    protscores.push(0);
                }
            }
            return protscores;
        } else if (nodeId in scoreMap) {
            return [(scoreMap[nodeId] - minScore) / (maxScore - minScore)];
        } else {
            return [];
        }
    }
    return protScorer;
}


export function protAggNone(protScores, forSort) {
    if (forSort) {
        return protAggMean(protScores);
    } else {
        return protScores;
    }
}


export function protAggMean(protScores) {
    return _.mean(protScores) || 0;
}

export function protAggMax(protScores) {
    return _.max(protScores) || 0;
}

export function protAggMedian(protScores) {
    // TODO: Should probably use a library, no need to sort.
    const N = protScores.length;
    if (N == 0) {
        return 0;
    }
    const sorted = _.sortBy(protScores);
    const mid = Math.floor(N / 2);
    if (N%2 == 0) {
        return (protScores[mid-1] + protScores[mid]) / 2;
    } else {
        return protScores[mid];
    }
}

export const protAggLookup = {
    scoreplot: protAggNone,
    mean: protAggMean,
    max: protAggMax,
    median: protAggMedian,
};

export type NodePathwayScorer = (node: string, scores: JobScoreData) => number;
export type NodeProtScorer = (node: string, scores: JobScoreData) => number[];
export interface PathwayScorers {
    pathScorer: NodePathwayScorer,
    protScorer: NodeProtScorer,
    protAgg: any,
}

export function makeScorers(config: PathwayConfig, pathwayData: PathwayData): PathwayScorers {
    return {
        pathScorer: makePwScorer(config.pathwayScoreType, config.febeQFilter),
        protScorer: makeProtScorer(pathwayData, config),
        protAgg: protAggLookup[config.protAgg],
    };
}

