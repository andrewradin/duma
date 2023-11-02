import * as React from 'react';

export const GeneProt = ({gene, prot, direction}) => {
    const style = {
        backgroundColor: '#eef',
        border: '1px solid #bbb',
        fontSize: '75%',
        marginLeft: '3px',
        padding: '2px',
    };
    const dirs = ['🡇', '🡆', '🡅']
    const dir = dirs[direction + 1];
    return (
        <span style={style}>
            <a target='_blank' href={'/42/protein/' + prot}>{gene}</a>{dir}
        </span>
    );
};
