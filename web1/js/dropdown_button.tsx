
import * as React from 'react';
import {useState, useEffect} from 'react';


/**
 * ReactElement for a element in a dropdown list.
 * item can optionally have an 'items' property, in which case this entry becomes
 * a SubDropdown displaying those nested items.
 */
const DropdownItem = ({item, onClick, parentBlur, ...eventProps}) => {
    if (!item.items) {
        return (<li tabIndex="0" onBlur={parentBlur} {...eventProps}>
                <a style={{cursor: 'pointer'}} onClick={() => onClick(item)}>{item.label}</a>
            </li>);
    } else {
        return (
            <li className={'dropdown-submenu'} {...eventProps}>
                <SubDropdown
                    text={item.label}
                    initItems={item.items}
                    onClick={onClick}
                    parentBlur={parentBlur}
                />
            </li>
        );
    }
};

const SubDropdown = ({text, initItems, onClick, classNames, parentBlur}) => {
    const [dropState, setDropState] = useState({ dropped: false, });
    const initLoaded = (typeof initItems == "function") ? false : true;

    const [items, setItems] = useState(initLoaded ? initItems : []);
    const [loading, setLoading] = useState(initLoaded);

    // Track whether the mouse is over us or one of our children.  This is used
    // for ignoring blur events related to switching focus to children.
    const [hover, setHover] = useState(false);

    const waitForLoad = async () => {
        const newItems = await initItems();
        setItems(newItems);
    };
    if (!parentBlur) {
        parentBlur = () => {};
    }

    const onDropClick = () => {
        setDropState({ dropped: !dropState.dropped, });
    };

    // We pass this around so that it gets invoked both by our own blur, and
    // by our children blur.
    const onBlur = (evt) => {
        if (!hover) {
            setDropState({ dropped: false, });
            parentBlur();
        }
    };

    let content = null;
    const topClasses = 'dropdown' + (dropState.dropped ? ' open' : '');
    if (dropState.dropped) {
        if (!initLoaded && !loading) {
            setLoading(true);
            waitForLoad();
        }
        const menuClasses = 'dropdown-menu';
        const innerMenu = (
            <ul className={menuClasses + ' ' + topClasses} style={{'display': 'block'}}>
                {items.map(item => (
                    <DropdownItem
                        item={item}
                        onClick={onClick}
                        parentBlur={onBlur}
                        onMouseEnter={() => setHover(true)}
                        onMouseLeave={() => setHover(false)}
                    />
                )}
            </ul>
        );
        content = innerMenu;
    }
    const buttonClasses = "dropdown-toggle " + classNames;

    return (
        <>
            <a style={{cursor:'pointer'}} tabIndex="0" className={buttonClasses} onClick={onDropClick} onBlur={onBlur}>
                {text} <span className='caret'></span>
            </a>
            {content}
        </>
    )
};

/**
 * A button that acts as a dropdown when clicked on.
 * text: What to display on the button.
 * items: A list of items to show, or an async function returning a list of items.
 *       Each items should have a "label", and an optional "items" field (for sub dropdowns).
 * onClick: Invoked with the selected "item" object when one is selected.
 */
export const DropdownButton = ({text, items, classNames, onClick}) => {
    const topClasses = 'dropdown';

    return (
        <div tabIndex="0" className={topClasses} style={{'display': 'inline-block'}}>
            <SubDropdown classNames={classNames} initItems={items} text={text} onClick={onClick} />
        </div>
    )
};
