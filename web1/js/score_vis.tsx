import * as React from 'react';
import {useState, useEffect} from 'react';
import * as ReactDOM from 'react-dom';
import * as _ from 'lodash';

import * as Score from './pathway_scoring';
import {PathwayNode} from './pathway_data';
import {JobScoreData} from './pathway_tree';

export const MiniScorePlot = ({scores, slices, color}) => {
    /**
     * Scores should be 0-1 normalized already.
     * It can be an array or a single value.
     */

    // Early-exit if we have a single score.
    if (_.isNumber(scores)) {
        return <FilledBox color={color} h={scores} />;
    } else if (scores.length == 1) {
        return <FilledBox color={color} h={scores[0]} />;
    }

    scores.sort((a, b) => { return b - a; });
    const N = scores.length;
    slices = Math.min(slices, N);

    const step = N / slices;
    const markStyle = {
        display: 'inline-block',
        backgroundColor: 'white',
        width: '2px',
        height: '100%',
    };
    const containerStyle = {
        border: '1px solid #555',
        display: 'inline-block',
        height: '1rem',
        lineHeight: 0,
        marginRight: '-1px',
        // These two are used to align marks and miniscoreplots, which
        // weirdly offset if using the default center align.
        verticalAlign: 'top',
        marginTop: '5px',
    }
    const els = [];
    for (let i = 0; i < slices; ++i) {
        const idx = Math.floor(i * step);
        const val = scores[idx];
        const h = val * 100;
        const fillStyle = {
            ...markStyle,
            background: `linear-gradient(to top, ${color} ${h}%, white ${h}%` ,
        };
        els.push((<span style={fillStyle}/>));
    }

    return <span style={containerStyle}>{els}</span>;
};

export const CollectionMarker = ({node, collection, color}) {
    const h = (node.id in collection.pathways) ? 1.0 : 0.0;
    return <FilledBox color={color} h={h} />;
}

interface AggScoreMarkerParms {
    node: PathwayNode,
    scores: JobScoreData[],
    scorers: Score.PathwayScorers,
}
export const AggScoreMarker = ({node, scores, scorers}: AggScoreMarkerParms) => {
    const {pathScorer, protScorer, protAgg} = scorers;
    const pwScores = [];
    const protScores = [];

    let weightSum = 0;
    for (const score of scores) {
        weightSum += score.weight;
    }

    for (const score of scores) {
        const weight = score.weight / weightSum;
        if (!score.pwToScore) {
            const curProtScores = protScorer(node.id, score);
            const agg = protAgg(curProtScores, true);
            protScores.push(agg * weight);
        } else {
            const h = pathScorer(node.id, score);
            if (h !== undefined) {
                pwScores.push(h * weight);
            }
        }
    }

    let pwAggEl = null;
    let protAggEl = null;
    // Sum used below because we're already normalizing by # of scores via
    // the weightSum above.
    if (pwScores.length > 0) {
        const h = _.sum(pwScores);
        pwAggEl = <FilledBox color={"#777"} h={h} />;
    }
    if (protScores.length > 0) {
        const h = _.sum(protScores);
        protAggEl = <FilledBox color={"#aaa"} h={h} />;
    }

    return (<>{pwAggEl}{protAggEl} </>);
};

interface ScoreMarkerParms {
    node: PathwayNode,
    score: JobScoreData,
    color: string,
    scorers: Score.PathwayScorers,
}
export const ScoreMarker = ({node, score, color, scorers}: ScoreMarkerParms) => {
    const {pathScorer, protScorer, protAgg} = scorers;
    if (!score.pwToScore) {
        const protScores = protScorer(node.id, score);
        const agg = protAgg(protScores);
        return <MiniScorePlot scores={agg} slices={8} color={color}/>);
    } else {
        const h = pathScorer(node.id, score);
        if (h === undefined) {
            return null;
        } else {
            return <FilledBox color={color} h={h} />;
        }
    }
};


export const FilledBox = ({color, h}) {
    const markStyle = {
        display: 'inline-block',
        backgroundColor: 'white',
        border: '1px solid #555',
        marginRight: '-1px',
        width: '6px',
        height: '1rem',
        // These two are used to align marks and miniscoreplots, which
        // weirdly offset if using the default center align.
        verticalAlign: 'top',
        marginTop: '5px',

    };
    const fillStyle = {
        ...markStyle,
        background: `linear-gradient(to top, ${color} ${h*100}%, white ${h*100}%` ,
    };
    return (<span style={fillStyle}/>);
}
