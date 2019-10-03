(function(root, factory) {
	if (typeof module === 'object' && module.exports) {
		var d3 = require('d3');
		module.exports = factory(d3);
	} else if(typeof define === 'function' && define.amd) {
		try {
			var d3 = require('d3');
		} catch (e) {
			d3 = root.d3;
		}

		d3.contextMenu = factory(d3);
		define([], function() {
			return d3.contextMenu;
		});
	} else if(root.d3) {
		root.d3.contextMenu = factory(root.d3);
	}
}(this,
	function (d3) {
		var utils = {
			noop: function () {},

			/**
			 * @param {*} value
			 * @returns {Boolean}
			 */
			isFn: function (value) {
				return typeof value === 'function';
			},

			/**
			 * @param {*} value
			 * @returns {Function}
			 */
			const: function (value) {
				return function () { return value; };
			},

			/**
			 * @param {Function|*} value
			 * @param {*} [fallback]
			 * @returns {Function}
			 */
			toFactory: function (value, fallback) {
				value = (value === undefined) ? fallback : value;
				return utils.isFn(value) ? value : utils.const(value);
			}
		};

		// global state for d3-context-menu
		var d3ContextMenu = null;

		var closeMenu = function () {
			// global state is populated if a menu is currently opened
			if (d3ContextMenu) {
				d3.select('.d3-context-menu').remove();
				d3.select('body').on('mousedown.d3-context-menu', null);
				d3ContextMenu.boundCloseCallback();
				d3ContextMenu = null;
			}
		};

		/**
		 * Calls API method (e.g. `close`) or
		 * returns handler function for the `contextmenu` event
		 * @param {Function|Array|String} menuItems
		 * @param {Function|Object} config
		 * @returns {?Function}
		 */
		return function (menuItems, config) {
			// allow for `d3.contextMenu('close');` calls
			// to programatically close the menu
			if (menuItems === 'close') {
				return closeMenu();
			}

			// for convenience, make `menuItems` a factory
			// and `config` an object
			menuItems = utils.toFactory(menuItems);

			if (utils.isFn(config)) {
				config = { onOpen: config };
			}
			else {
				config = config || {};
			}

			// resolve config
			var openCallback = config.onOpen || utils.noop;
			var closeCallback = config.onClose || utils.noop;
			var positionFactory = utils.toFactory(config.position);
			var themeFactory = utils.toFactory(config.theme, 'd3-context-menu-theme');

			/**
			 * Context menu event handler
			 * @param {*} data
			 * @param {Number} index
			 */
			return function (data, index) {
				var element = this;

				// close any menu that's already opened
				closeMenu();

				// store close callback already bound to the correct args and scope
				d3ContextMenu = {
					boundCloseCallback: closeCallback.bind(element, data, index)
				};

				// create the div element that will hold the context menu
				d3.selectAll('.d3-context-menu').data([1])
					.enter()
					.append('div')
					.attr('class', 'd3-context-menu ' + themeFactory.bind(element)(data, index));

				// close menu on mousedown outside
				d3.select('body').on('mousedown.d3-context-menu', closeMenu);
				d3.select('body').on('click.d3-context-menu', closeMenu);

				var parent = d3.selectAll('.d3-context-menu')
					.on('contextmenu', function() {
						closeMenu();
						d3.event.preventDefault();
						d3.event.stopPropagation();
					})
					.append('ul');

				parent.call(createNestedMenu, element);

				// the openCallback allows an action to fire before the menu is displayed
				// an example usage would be closing a tooltip
				if (openCallback.bind(element)(data, index) === false) {
					return;
				}

				// get position
				var position = positionFactory.bind(element)(data, index);

				// display context menu
				d3.select('.d3-context-menu')
					.style('left', (position ? position.left : d3.event.pageX - 2) + 'px')
					.style('top', (position ? position.top : d3.event.pageY - 2) + 'px')
					.style('display', 'block');

				d3.event.preventDefault();
				d3.event.stopPropagation();


				function createNestedMenu(parent, root, depth = 0) {
					var resolve = function (value) {
						return utils.toFactory(value).call(root, data, index);
					};

					parent.selectAll('li')
					.data(function (d) {
							var baseData = depth === 0 ? menuItems : d.children;
							return resolve(baseData);
						})
						.enter()
						.append('li')
						.each(function (d) {
							// get value of each data
							var isDivider = !!resolve(d.divider);
							var isDisabled = !!resolve(d.disabled);
							var hasChildren = !!resolve(d.children);
							var hasAction = !!d.action;
							var text = isDivider ? '<hr>' : resolve(d.title);

							var listItem = d3.select(this)
								.classed('is-divider', isDivider)
								.classed('is-disabled', isDisabled)
								.classed('is-header', !hasChildren && !hasAction)
								.classed('is-parent', hasChildren)
								.html(text)
								.on('click', function () {
									// do nothing if disabled or no action
									if (isDisabled || !hasAction) return;

									d.action.call(root, data, index);
									closeMenu();
								});

							if (hasChildren) {
								// create children(`next parent`) and call recursive
								var children = listItem.append('ul').classed('is-children', true);
								createNestedMenu(children, root, ++depth)
							}
						});
				}
			};
		};
	}
));


function PtN2Diagram(parentDiv, modelJSON) {
    var model = new ModelData(modelJSON);

    setD3ContentDiv();
    var svg = d3.select("#svgId");

    // TODO: Get rid of all these after refactoring ///////////////
    var root = model.root;
    var conns = model.conns;
    var abs2prom = model.abs2prom;
    ///////////////////////////////////////////////////////////////

    var svgStyleElement = document.createElement("style");
    var showPath = false; //default off

    var DEFAULT_TRANSITION_START_DELAY = 100;
    var transitionStartDelay = DEFAULT_TRANSITION_START_DELAY;

    //N^2 vars
    var backButtonHistory = [], forwardButtonHistory = [];
    var chosenCollapseDepth = -1;
    var updateRecomputesAutoComplete = true; //default

    var tooltip = d3.select("body").append("div").attr("class", "tool-tip")
        .style("position", "absolute")
        .style("visibility", "hidden");

    mouseOverOnDiagN2 = MouseoverOnDiagN2;
    mouseOverOffDiagN2 = MouseoverOffDiagN2;
    mouseClickN2 = MouseClickN2;
    mouseOutN2 = MouseoutN2;

    CreateDomLayout();
    CreateToolbar();

    parentDiv.querySelector("#svgId").appendChild(svgStyleElement);
    UpdateSvgCss(svgStyleElement, N2SVGLayout.fontSizePx);

    arrowMarker = svg.append("svg:defs").append("svg:marker");

    arrowMarker
        .attr("id", "arrow")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 5)
        .attr("refY", 0)
        .attr("markerWidth", 1)
        .attr("markerHeight", 1)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("class", "arrowHead");

    setN2Group();
    var pTreeGroup = svg.append("g").attr("id", "tree"); // id given just so it is easier to see in Chrome dev tools when debugging
    var pSolverTreeGroup = svg.append("g").attr("id", "solver_tree");

    var n2BackgroundRect = n2Group.append("rect")
        .attr("class", "background")
        .attr("width", WIDTH_N2_PX)
        .attr("height", N2SVGLayout.heightPx);

    setN2ElementsGroup();

    var zoomedElement0 = model.root;
    var lastRightClickedElement = model.root;

    var layout = new N2SVGLayout(model, model.root);
    // TODO: Get rid of all these after refactoring ///////////////

    d3NodesArray = layout.zoomedNodes;
    d3RightTextNodesArrayZoomed = layout.visibleNodes;

    d3SolverNodesArray = layout.zoomedSolverNodes;
    d3SolverRightTextNodesArrayZoomed = layout.visibleSolverNodes;
    ///////////////////////////////////////////////////////////////

    ComputeLayout();
    ComputeConnections();

    matrix = new N2Matrix(layout.visibleNodes);

    var collapseDepthElement = parentDiv.querySelector("#idCollapseDepthDiv");
    for (var i = 2; i <= model.maxDepth; ++i) {
        var option = document.createElement("span");
        option.className = "fakeLink";
        option.id = "idCollapseDepthOption" + i + "";
        option.innerHTML = "" + i + "";
        var f = function (idx) {
            return function () {
                CollapseToDepthSelectChange(idx);
            };
        }(i);
        option.onclick = f;
        collapseDepthElement.appendChild(option);
    }

    var menu = [{
                    title: 'Collapse',
                    action: function (data, index) {
                        CollapseFromRightClick(data, this);
                    }
                },
                {
                    title: 'Item #2',
                },
               ];


    Update(false);
    SetupLegend(d3, d3ContentDiv);

    function Update(computeNewTreeLayout = true) {
        parentDiv.querySelector("#currentPathId").innerHTML = "PATH: root" + ((zoomedElement.parent) ? "." : "") + zoomedElement.absPathName;

        parentDiv.querySelector("#backButtonId").disabled = (backButtonHistory.length == 0) ? "disabled" : false;
        parentDiv.querySelector("#forwardButtonId").disabled = (forwardButtonHistory.length == 0) ? "disabled" : false;
        parentDiv.querySelector("#upOneLevelButtonId").disabled = (zoomedElement === root) ? "disabled" : false;
        parentDiv.querySelector("#returnToRootButtonId").disabled = (zoomedElement === root) ? "disabled" : false;

        // Compute the new tree layout.
        if (computeNewTreeLayout) {
            layout = new N2SVGLayout(model, zoomedElement);
            // TODO: Get rid of all these after refactoring ///////////////
            d3NodesArray = layout.zoomedNodes;
            d3RightTextNodesArrayZoomed = layout.visibleNodes;

            d3SolverNodesArray = layout.zoomedSolverNodes;
            d3SolverRightTextNodesArrayZoomed = layout.visibleSolverNodes;
            ///////////////////////////////////////////////////////////////

            ComputeLayout();

            matrix = new N2Matrix(layout.visibleNodes);
        }

        for (var i = 2; i <= model.maxDepth; ++i) {
            parentDiv.querySelector("#idCollapseDepthOption" + i + "").style.display = (i <= zoomedElement.depth) ? "none" : "block";
        }

        if (xScalerPTree0 != null) {//not first run.. store previous
            kx0 = kx;
            ky0 = ky;
            xScalerPTree0 = xScalerPTree.copy();
            yScalerPTree0 = yScalerPTree.copy();

            kxSolver0 = kxSolver;
            kySolver0 = kySolver;
            xScalerPSolverTree0 = xScalerPSolverTree.copy();
            yScalerPSolverTree0 = yScalerPSolverTree.copy();
        }

        kx = (zoomedElement.x ? layout.widthPTreePx - N2SVGLayout.parentNodeWidthPx : layout.widthPTreePx) / (1 - zoomedElement.x);
        ky = N2SVGLayout.heightPx / zoomedElement.height;
        xScalerPTree.domain([zoomedElement.x, 1]).range([zoomedElement.x ? N2SVGLayout.parentNodeWidthPx : 0, layout.widthPTreePx]);
        yScalerPTree.domain([zoomedElement.y, zoomedElement.y + zoomedElement.height]).range([0, N2SVGLayout.heightPx]);

        kxSolver = (zoomedElement.xSolver ? layout.widthPSolverTreePx - N2SVGLayout.parentNodeWidthPx : layout.widthPSolverTreePx) / (1 - zoomedElement.xSolver);
        kySolver = N2SVGLayout.heightPx / zoomedElement.heightSolver;
        xScalerPSolverTree.domain([zoomedElement.xSolver, 1]).range([zoomedElement.xSolver ? N2SVGLayout.parentNodeWidthPx : 0, layout.widthPSolverTreePx]);
        yScalerPSolverTree.domain([zoomedElement.ySolver, zoomedElement.ySolver + zoomedElement.heightSolver]).range([0, N2SVGLayout.heightPx]);

        if (xScalerPTree0 == null) { //first run.. duplicate
            kx0 = kx;
            ky0 = ky;
            xScalerPTree0 = xScalerPTree.copy();
            yScalerPTree0 = yScalerPTree.copy();

            kxSolver0 = kxSolver;
            kySolver0 = kySolver;
            xScalerPSolverTree0 = xScalerPSolverTree.copy();
            yScalerPSolverTree0 = yScalerPSolverTree.copy();

            //Update svg dimensions before ComputeLayout() changes layout.widthPTreePx
            svgDiv.style("width", (layout.widthPTreePx + PTREE_N2_GAP_PX + WIDTH_N2_PX + layout.widthPSolverTreePx + 2 * SVG_MARGIN + PTREE_N2_GAP_PX) + "px")
                .style("height", (N2SVGLayout.heightPx + 2 * SVG_MARGIN) + "px");
            svg.attr("width", layout.widthPTreePx + PTREE_N2_GAP_PX + WIDTH_N2_PX + layout.widthPSolverTreePx + 2 * SVG_MARGIN + PTREE_N2_GAP_PX)
                .attr("height", N2SVGLayout.heightPx + 2 * SVG_MARGIN);

            n2Group.attr("transform", "translate(" + (layout.widthPTreePx + PTREE_N2_GAP_PX + SVG_MARGIN) + "," + SVG_MARGIN + ")");
            pTreeGroup.attr("transform", "translate(" + SVG_MARGIN + "," + SVG_MARGIN + ")");

            pSolverTreeGroup.attr("transform", "translate(" + (layout.widthPTreePx + PTREE_N2_GAP_PX + WIDTH_N2_PX + SVG_MARGIN + PTREE_N2_GAP_PX) + "," + SVG_MARGIN + ")");
        }

        sharedTransition = d3.transition().duration(TRANSITION_DURATION).delay(transitionStartDelay); //do this after intense computation
        transitionStartDelay = DEFAULT_TRANSITION_START_DELAY;

        //Update svg dimensions with transition after ComputeLayout() changes layout.widthPTreePx
        svgDiv.transition(sharedTransition).style("width", (layout.widthPTreePx + PTREE_N2_GAP_PX + WIDTH_N2_PX + layout.widthPSolverTreePx + 2 * SVG_MARGIN + PTREE_N2_GAP_PX) + "px")
            .style("height", (N2SVGLayout.heightPx + 2 * SVG_MARGIN) + "px");
        svg.transition(sharedTransition).attr("width", layout.widthPTreePx + PTREE_N2_GAP_PX + WIDTH_N2_PX + layout.widthPSolverTreePx + 2 * SVG_MARGIN + PTREE_N2_GAP_PX)
            .attr("height", N2SVGLayout.heightPx + 2 * SVG_MARGIN);

        n2Group.transition(sharedTransition).attr("transform", "translate(" + (layout.widthPTreePx + PTREE_N2_GAP_PX + SVG_MARGIN) + "," + SVG_MARGIN + ")");
        pTreeGroup.transition(sharedTransition).attr("transform", "translate(" + SVG_MARGIN + "," + SVG_MARGIN + ")");
        n2BackgroundRect.transition(sharedTransition).attr("width", WIDTH_N2_PX).attr("height", HEIGHT_PX);

        pSolverTreeGroup.transition(sharedTransition).attr("transform", "translate(" + (layout.widthPTreePx + PTREE_N2_GAP_PX + WIDTH_N2_PX + SVG_MARGIN + PTREE_N2_GAP_PX) + "," + SVG_MARGIN + ")");

        var sel = pTreeGroup.selectAll(".partition_group")
            .data(d3NodesArray, function (d) {
                return d.id;
            });

        var nodeEnter = sel.enter().append("svg:g")
            .attr("class", function (d) {
                return "partition_group " + GetClass(d);
            })
            .attr("transform", function (d) {
                return "translate(" + xScalerPTree0(d.x0) + "," + yScalerPTree0(d.y0) + ")";
            })
            .on("click", function (d) { LeftClick(d, this); })
            .on("contextmenu", d3.contextMenu(menu, {
				theme: function () {
					return 'd3-context-menu-theme';
				},
				onOpen: function (data, index) {
					console.log('Menu Opened!', 'element:', this, 'data:', data, 'index:', index);
				},
				onClose: function (data, index) {
					console.log('Menu Closed!', 'element:', this, 'data:', data, 'index:', index);
				},
				position: function (data, index) {
					var position = d3.mouse(document.body)
					console.log('position!', position);
					return {
						    left: position[0],
						    top: position[1]
						}
				}
			}))


            .on("mouseover", function (d) {
                if (abs2prom != undefined) {
                    if (d.type == "param" || d.type == "unconnected_param") {
                        return tooltip.text(abs2prom.input[d.absPathName])
                            .style("visibility", "visible");
                    }
                    if (d.type == "unknown") {
                        return tooltip.text(abs2prom.output[d.absPathName])
                            .style("visibility", "visible");
                    }
                }
            })
            .on("mouseleave", function (d) {
                if (abs2prom != undefined) {
                    return tooltip.style("visibility", "hidden");
                }
            })
            .on("mousemove", function () {
                if (abs2prom != undefined) {
                    return tooltip.style("top", (d3.event.pageY - 30) + "px")
                        .style("left", (d3.event.pageX + 5) + "px");
                }
            });

        nodeEnter.append("svg:rect")
            .attr("width", function (d) {
                return d.width0 * kx0;//0;//
            })
            .attr("height", function (d) {
                return d.height0 * ky0;
            });

        nodeEnter.append("svg:text")
            .attr("dy", ".35em")
            //.attr("text-anchor", "end")
            .attr("transform", function (d) {
                var anchorX = d.width0 * kx0 - N2SVGLayout.rightTextMarginPx;
                return "translate(" + anchorX + "," + d.height0 * ky0 / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < zoomedElement.depth) return 0;
                return d.textOpacity0;
            })
            .text(layout.getText);

        var nodeUpdate = nodeEnter.merge(sel).transition(sharedTransition)
            .attr("class", function (d) {
                return "partition_group " + GetClass(d);
            })
            .attr("transform", function (d) {
                return "translate(" + xScalerPTree(d.x) + "," + yScalerPTree(d.y) + ")";
            });

        nodeUpdate.select("rect")
            .attr("width", function (d) {
                return d.width * kx;
            })
            .attr("height", function (d) {
                return d.height * ky;
            });

        nodeUpdate.select("text")
            .attr("transform", function (d) {
                var anchorX = d.width * kx - N2SVGLayout.rightTextMarginPx;
                return "translate(" + anchorX + "," + d.height * ky / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(layout.getText);


        // Transition exiting nodes to the parent's new position.
        var nodeExit = sel.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + xScalerPTree(d.x) + "," + yScalerPTree(d.y) + ")";
            })
            .remove();

        nodeExit.select("rect")
            .attr("width", function (d) {
                return d.width * kx;//0;//
            })
            .attr("height", function (d) {
                return d.height * ky;
            });

        nodeExit.select("text")
            .attr("transform", function (d) {
                var anchorX = d.width * kx - N2SVGLayout.rightTextMarginPx;
                return "translate(" + anchorX + "," + d.height * ky / 2 + ")";
                //return "translate(8," + d.height * ky / 2 + ")";
            })
            .style("opacity", 0);


        var selSolver = pSolverTreeGroup.selectAll(".solver_group")
            .data(d3SolverNodesArray, function (d) {
                return d.id;
            });

        function getSolverClass(showLinearSolverNames, linear_solver_name, nonlinear_solver_name) {
            if (showLinearSolverNames) {
                if (linearSolverNames.indexOf(linear_solver_name) >= 0) {
                    solver_class = linearSolverClasses[linear_solver_name]
                } else {
                    solver_class = linearSolverClasses["other"]; // user must have defined their own solver that we do not know about
                }
            } else {
                if (nonLinearSolverNames.indexOf(nonlinear_solver_name) >= 0) {
                    solver_class = nonLinearSolverClasses[nonlinear_solver_name]
                } else {
                    solver_class = nonLinearSolverClasses["other"]; // user must have defined their own solver that we do not know about
                }
            }
            return solver_class;
        }

        var nodeSolverEnter = selSolver.enter().append("svg:g")
            .attr("class", function (d) {
                solver_class = getSolverClass(N2SVGLayout.showLinearSolverNames, d.linear_solver, d.nonlinear_solver);
                return solver_class + " " + "solver_group " + GetClass(d);
            })
            .attr("transform", function (d) {
                x = 1.0 - d.xSolver0 - d.widthSolver0; // The magic for reversing the blocks on the right side
                // The solver tree goes from the root on the right and expands to the left
                return "translate(" + xScalerPSolverTree0(x) + "," + yScalerPSolverTree0(d.ySolver0) + ")";
            })
            .on("click", function (d) { LeftClick(d, this); })
            .on("contextmenu", function (d) { RightClick(d, this); })
            .on("mouseover", function (d) {
                if (abs2prom != undefined) {
                    if (d.type == "param" || d.type == "unconnected_param") {
                        return tooltip.text(abs2prom.input[d.absPathName])
                            .style("visibility", "visible");
                    }
                    if (d.type == "unknown") {
                        return tooltip.text(abs2prom.output[d.absPathName])
                            .style("visibility", "visible");
                    }
                }
            })
            .on("mouseleave", function (d) {
                if (abs2prom != undefined) {
                    return tooltip.style("visibility", "hidden");
                }
            })
            .on("mousemove", function () {
                if (abs2prom != undefined) {
                    return tooltip.style("top", (d3.event.pageY - 30) + "px")
                        .style("left", (d3.event.pageX + 5) + "px");
                }
            });

        nodeSolverEnter.append("svg:rect")
            .attr("width", function (d) {
                return d.widthSolver0 * kxSolver0;//0;//
            })
            .attr("height", function (d) {
                return d.heightSolver0 * kySolver0;
            });

        nodeSolverEnter.append("svg:text")
            .attr("dy", ".35em")
            .attr("transform", function (d) {
                var anchorX = d.widthSolver0 * kxSolver0 - N2SVGLayout.rightTextMarginPx;
                return "translate(" + anchorX + "," + d.heightSolver0 * kySolver0 / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < zoomedElement.depth) return 0;
                return d.textOpacity0;
            })
            .text(layout.getSolverText);

        var nodeSolverUpdate = nodeSolverEnter.merge(selSolver).transition(sharedTransition)
            .attr("class", function (d) {
                solver_class = getSolverClass(N2SVGLayout.showLinearSolverNames, d.linear_solver, d.nonlinear_solver);
                return solver_class + " " + "solver_group " + GetClass(d);
            })
            .attr("transform", function (d) {
                x = 1.0 - d.xSolver - d.widthSolver; // The magic for reversing the blocks on the right side
                return "translate(" + xScalerPSolverTree(x) + "," + yScalerPSolverTree(d.ySolver) + ")";
            });

        nodeSolverUpdate.select("rect")
            .attr("width", function (d) {
                return d.widthSolver * kxSolver;
            })
            .attr("height", function (d) {
                return d.heightSolver * kySolver;
            });

        nodeSolverUpdate.select("text")
            .attr("transform", function (d) {
                var anchorX = d.widthSolver * kxSolver - N2SVGLayout.rightTextMarginPx;
                return "translate(" + anchorX + "," + d.heightSolver * kySolver / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(layout.getSolverText);


        // Transition exiting nodes to the parent's new position.
        var nodeSolverExit = selSolver.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + xScalerPSolverTree(d.xSolver) + "," + yScalerPSolverTree(d.ySolver) + ")";
            })
            .remove();

        nodeSolverExit.select("rect")
            .attr("width", function (d) {
                return d.widthSolver * kxSolver;//0;//
            })
            .attr("height", function (d) {
                return d.heightSolver * kySolver;
            });

        nodeSolverExit.select("text")
            .attr("transform", function (d) {
                var anchorX = d.widthSolver * kxSolver - N2SVGLayout.rightTextMarginPx;
                return "translate(" + anchorX + "," + d.heightSolver * kySolver / 2 + ")";
            })
            .style("opacity", 0);


        ClearArrowsAndConnects()
        // DrawMatrix();
        matrix.draw();
    }

    updateFunc = Update;

    function ClearArrows() {
        n2Group.selectAll("[class^=n2_hover_elements]").remove();
    }

    function ClearArrowsAndConnects() {
        ClearArrows();
        // newConnsDict = {};
        // PrintConnects();
    }

    function ComputeLayout() {
        var autoCompleteSetNames = {}, autoCompleteSetPathNames = {};

        function PopulateAutoCompleteList(d) {
            if (d.children && !d.isMinimized) { //depth first, dont go into minimized children
                for (var i = 0; i < d.children.length; ++i) {
                    PopulateAutoCompleteList(d.children[i]);
                }
            }
            if (d === zoomedElement) return;

            var n = d.name;
            if (d.splitByColon && d.children && d.children.length > 0) n += ":";
            if ((d.type !== "param" && d.type !== "unconnected_param") && d.type !== "unknown") n += ".";
            var namesToAdd = [n];

            if (d.splitByColon) namesToAdd.push(d.colonName + ((d.children && d.children.length > 0) ? ":" : ""));

            namesToAdd.forEach(function (name) {
                if (!autoCompleteSetNames.hasOwnProperty(name)) {
                    autoCompleteSetNames[name] = true;
                    autoCompleteListNames.push(name);
                }
            });

            var localPathName = (zoomedElement === root) ? d.absPathName : d.absPathName.slice(zoomedElement.absPathName.length + 1);
            if (!autoCompleteSetPathNames.hasOwnProperty(localPathName)) {
                autoCompleteSetPathNames[localPathName] = true;
                autoCompleteListPathNames.push(localPathName);
            }
        }

        if (updateRecomputesAutoComplete) {
            autoCompleteListNames = [];
            autoCompleteListPathNames = [];
            PopulateAutoCompleteList(zoomedElement);
        }
        updateRecomputesAutoComplete = true; //default

        enterIndex = exitIndex = 0;
        if (lastClickWasLeft) { //left click
            if (leftClickIsForward) {
                exitIndex = lastLeftClickedElement.rootIndex - zoomedElement0.rootIndex;
            }
            else {
                enterIndex = zoomedElement0.rootIndex - lastLeftClickedElement.rootIndex;
            }
        }

        // Not sure if necessary...
        // textWidthGroup.remove();
    }

    var lastLeftClickedEle;
    var lastRightClickedEle;
    var lastRightClickedObj;

    //right click => collapse
    function RightClick(d, ele) {
        var position = d3.mouse(this);
        d3.select('#context_menu_sys')
          .style('position', 'absolute')
          .style('left', position[0] + "px")
          .style('top', position[1] + "px")
          .style('display', 'block');

        d3.contextMenu(menu, {
        });

        var e = d3.event;
        e.preventDefault();
    }
    function CollapseFromRightClick(d, ele) {
        var e = d3.event;
        lastLeftClickedEle = d;
        lastRightClickedObj = d;
        lastRightClickedEle = ele;
        e.preventDefault();
        collapse();
    }

    var menu = document.querySelector('#context-menu');
    var menuState = 0;
    var contextMenuActive = "context-menu--active";

    function collapse() {
        var d = lastLeftClickedEle;
        if (!d.children) return;
        if (d.depth > zoomedElement.depth) { //dont allow minimizing on root node
            lastRightClickedElement = d;
            FindRootOfChangeFunction = FindRootOfChangeForRightClick;
            TRANSITION_DURATION = TRANSITION_DURATION_FAST;
            lastClickWasLeft = false;
            Toggle(d);
            Update();
        }
    }

    function SetupLeftClick(d) {
        lastLeftClickedElement = d;
        lastClickWasLeft = true;
        if (lastLeftClickedElement.depth > zoomedElement.depth) {
            leftClickIsForward = true; //forward
        }
        else if (lastLeftClickedElement.depth < zoomedElement.depth) {
            leftClickIsForward = false; //backwards
        }
        zoomedElement0 = zoomedElement;
        zoomedElement = d;
        TRANSITION_DURATION = TRANSITION_DURATION_FAST;
    }

    //left click => navigate
    function LeftClick(d, ele) {
        if (!d.children) return;
        if (d3.event.button != 0) return;
        backButtonHistory.push({ "el": zoomedElement });
        forwardButtonHistory = [];
        SetupLeftClick(d);
        Update();
        d3.event.preventDefault();
        d3.event.stopPropagation();
    }

    function BackButtonPressed() {
        if (backButtonHistory.length == 0) return;
        var d = backButtonHistory.pop().el;
        parentDiv.querySelector("#backButtonId").disabled = (backButtonHistory.length == 0) ? "disabled" : false;
        for (var obj = d; obj != null; obj = obj.parent) { //make sure history item is not minimized
            if (obj.isMinimized) return;
        }
        forwardButtonHistory.push({ "el": zoomedElement });
        SetupLeftClick(d);
        Update();
    }

    function ForwardButtonPressed() {
        if (forwardButtonHistory.length == 0) return;
        var d = forwardButtonHistory.pop().el;
        parentDiv.querySelector("#forwardButtonId").disabled = (forwardButtonHistory.length == 0) ? "disabled" : false;
        for (var obj = d; obj != null; obj = obj.parent) { //make sure history item is not minimized
            if (obj.isMinimized) return;
        }
        backButtonHistory.push({ "el": zoomedElement });
        SetupLeftClick(d);
        Update();
    }

    function GetClass(d) {
        if (d.isMinimized) return "minimized";
        if (d.type === "param") {
            if (d.children && d.children.length > 0) return "param_group";
            return "param";
        }
        if (d.type === "unconnected_param") {
            if (d.children && d.children.length > 0) return "param_group";
            return "unconnected_param"
        }
        if (d.type === "unknown") {
            if (d.children && d.children.length > 0) return "unknown_group";
            if (d.implicit) return "unknown_implicit";
            return "unknown";
        }
        if (d.type === "root") return "subsystem";
        if (d.type === "subsystem") {
            if (d.subsystem_type === "component") return "component";
            return "subsystem";
        }
        alert("class not found");
    }

    function Toggle(d) {

        if (d.isMinimized)
            d.isMinimized = false;
        else
            d.isMinimized = true;
    }

    function ComputeConnections() {
        function GetObjectInTree(d, nameArray, nameIndex) {
            if (nameArray.length == nameIndex) {
                return d;
            }
            if (!d.children) {
                return null;
            }

            for (var i = 0; i < d.children.length; ++i) {
                if (d.children[i].name === nameArray[nameIndex]) {
                    return GetObjectInTree(d.children[i], nameArray, nameIndex + 1);
                }
                else {
                    var numNames = d.children[i].name.split(":").length;
                    if (numNames >= 2 && nameIndex + numNames <= nameArray.length) {
                        var mergedName = nameArray[nameIndex];
                        for (var j = 1; j < numNames; ++j) {
                            mergedName += ":" + nameArray[nameIndex + j];
                        }
                        if (d.children[i].name === mergedName) {
                            return GetObjectInTree(d.children[i], nameArray, nameIndex + numNames);
                        }
                    }
                }
            }
            return null;
        }

        function AddLeaves(d, objArray) {
            if (d.type !== "param" && d.type !== "unconnected_param") {
                objArray.push(d);
            }
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    AddLeaves(d.children[i], objArray);
                }
            }
        }

        function ClearConnections(d) {
            d.targetsParamView = new Set();
            d.targetsHideParams = new Set();

            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    ClearConnections(d.children[i]);
                }
            }
        }

        ClearConnections(root);

        var sys_pathnames = model.sys_pathnames_list;

        for (var i = 0; i < conns.length; ++i) {
            var srcSplitArray = conns[i].src.split(/\.|:/);
            var srcObj = GetObjectInTree(root, srcSplitArray, 0);
            if (srcObj == null) {
                alert("error: cannot find connection source " + conns[i].src);
                return;
            }
            var srcObjArray = [srcObj];
            if (srcObj.type !== "unknown") { //source obj must be unknown
                alert("error: there is a source that is not an unknown.");
                return;
            }
            if (srcObj.children) { //source obj must be unknown
                alert("error: there is a source that has children.");
                return;
            }
            for (var obj = srcObj.parent; obj != null; obj = obj.parent) {
                srcObjArray.push(obj);
            }

            var tgtSplitArray = conns[i].tgt.split(/\.|:/);
            var tgtObj = GetObjectInTree(root, tgtSplitArray, 0);
            if (tgtObj == null) {
                alert("error: cannot find connection target " + conns[i].tgt);
                return;
            }
            var tgtObjArrayParamView = [tgtObj];
            var tgtObjArrayHideParams = [tgtObj];
            if (tgtObj.type !== "param" && tgtObj.type !== "unconnected_param") { //target obj must be a param
                alert("error: there is a target that is NOT a param.");
                return;
            }
            if (tgtObj.children) {
                alert("error: there is a target that has children.");
                return;
            }
            AddLeaves(tgtObj.parentComponent, tgtObjArrayHideParams); //contaminate
            for (var obj = tgtObj.parent; obj != null; obj = obj.parent) {
                tgtObjArrayParamView.push(obj);
                tgtObjArrayHideParams.push(obj);
            }


            for (var j = 0; j < srcObjArray.length; ++j) {
                if (!srcObjArray[j].hasOwnProperty('targetsParamView')) srcObjArray[j].targetsParamView = new Set();
                if (!srcObjArray[j].hasOwnProperty('targetsHideParams')) srcObjArray[j].targetsHideParams = new Set();

                tgtObjArrayParamView.forEach(item => srcObjArray[j].targetsParamView.add(item));
                tgtObjArrayHideParams.forEach(item => srcObjArray[j].targetsHideParams.add(item));
            }

            var cycleArrowsArray = [];
            if (conns[i].cycle_arrows && conns[i].cycle_arrows.length > 0) {
                var cycleArrows = conns[i].cycle_arrows;
                for (var j = 0; j < cycleArrows.length; ++j) {
                    if (cycleArrows[j].length != 2) {
                        alert("error: cycleArrowsSplitArray length not 2, got " + cycleArrows[j].length +
                            ": " + cycleArrows[j]);
                        return;
                    }

                    var src_pathname = sys_pathnames[cycleArrows[j][0]];
                    var tgt_pathname = sys_pathnames[cycleArrows[j][1]];

                    var splitArray = src_pathname.split(/\.|:/);
                    var arrowBeginObj = GetObjectInTree(root, splitArray, 0);
                    if (arrowBeginObj == null) {
                        alert("error: cannot find cycle arrows begin object " + src_pathname);
                        return;
                    }
                    splitArray = tgt_pathname.split(/\.|:/);
                    var arrowEndObj = GetObjectInTree(root, splitArray, 0);
                    if (arrowEndObj == null) {
                        alert("error: cannot find cycle arrows end object " + tgt_pathname);
                        return;
                    }
                    cycleArrowsArray.push({ "begin": arrowBeginObj, "end": arrowEndObj });
                }
            }
            if (cycleArrowsArray.length > 0) {
                if (!tgtObj.parent.hasOwnProperty("cycleArrows")) {
                    tgtObj.parent.cycleArrows = [];
                }
                tgtObj.parent.cycleArrows.push({ "src": srcObj, "arrows": cycleArrowsArray });
            }

        }
    }

    function FindRootOfChangeForRightClick(d) {
        return lastRightClickedElement;
    }

    function FindRootOfChangeForCollapseDepth(d) {
        for (var obj = d; obj != null; obj = obj.parent) { //make sure history item is not minimized
            if (obj.depth == chosenCollapseDepth) return obj;
        }
        return d;
    }

    function FindRootOfChangeForCollapseUncollapseOutputs(d) {
        return (d.hasOwnProperty("parentComponent")) ? d.parentComponent : d;
    }

    function MouseoverOffDiagN2(d) {
        function GetObjectsInChildrenWithCycleArrows(d, arr) {
            if (d.cycleArrows) {
                arr.push(d);
            }
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    GetObjectsInChildrenWithCycleArrows(d.children[i], arr);
                }
            }
        }
        function GetObjectsWithCycleArrows(d, arr) {
            for (var obj = d.parent; obj != null; obj = obj.parent) { //start with parent.. the children will get the current object to avoid duplicates
                if (obj.cycleArrows) {
                    arr.push(obj);
                }
            }
            GetObjectsInChildrenWithCycleArrows(d, arr);
        }

        function HasObjectInChildren(d, toMatchObj) {
            if (d === toMatchObj) {
                return true;
            }
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    if (HasObjectInChildren(d.children[i], toMatchObj)) {
                        return true;
                    }
                }
            }
            return false;
        }
        function HasObject(d, toMatchObj) {
            for (var obj = d; obj != null; obj = obj.parent) {
                if (obj === toMatchObj) {
                    return true;
                }
            }
            return HasObjectInChildren(d, toMatchObj);
        }

        var lineWidth = Math.min(5, n2Dx * .5, n2Dy * .5);
        arrowMarker.attr("markerWidth", lineWidth * .4)
            .attr("markerHeight", lineWidth * .4);
        var src = d3RightTextNodesArrayZoomed[d.row];
        var tgt = d3RightTextNodesArrayZoomed[d.col];
        var boxEnd = d3RightTextNodesArrayZoomedBoxInfo[d.col];

        new N2Arrow({
            start: { col: d.row, row: d.row },
            end: { col: d.col, row: d.col },
            color: RED_ARROW_COLOR,
            width: lineWidth
        });

        if (d.row > d.col) {
            var targetsWithCycleArrows = [];
            GetObjectsWithCycleArrows(tgt, targetsWithCycleArrows);
            for (var ti = 0; ti < targetsWithCycleArrows.length; ++ti) {
                var arrows = targetsWithCycleArrows[ti].cycleArrows;
                for (var ai = 0; ai < arrows.length; ++ai) {
                    if (HasObject(src, arrows[ai].src)) {
                        var correspondingSrcArrows = arrows[ai].arrows;
                        for (var si = 0; si < correspondingSrcArrows.length; ++si) {
                            var beginObj = correspondingSrcArrows[si].begin;
                            var endObj = correspondingSrcArrows[si].end;
                            //alert(beginObj.name + "->" + endObj.name);
                            var firstBeginIndex = -1, firstEndIndex = -1;

                            //find first begin index
                            for (var mi = 0; mi < d3RightTextNodesArrayZoomed.length; ++mi) {
                                var rtNode = d3RightTextNodesArrayZoomed[mi];
                                if (HasObject(rtNode, beginObj)) {
                                    firstBeginIndex = mi;
                                    break;
                                }
                            }
                            if (firstBeginIndex == -1) {
                                alert("error: first begin index not found");
                                return;
                            }

                            //find first end index
                            for (var mi = 0; mi < d3RightTextNodesArrayZoomed.length; ++mi) {
                                var rtNode = d3RightTextNodesArrayZoomed[mi];
                                if (HasObject(rtNode, endObj)) {
                                    firstEndIndex = mi;
                                    break;
                                }
                            }
                            if (firstEndIndex == -1) {
                                alert("error: first end index not found");
                                return;
                            }

                            if (firstBeginIndex != firstEndIndex) {
                                DrawArrowsParamView(firstBeginIndex, firstEndIndex);
                            }
                        }
                    }
                }
            }
        }

        var leftTextWidthR = d3RightTextNodesArrayZoomed[d.row].nameWidthPx,
            leftTextWidthC = d3RightTextNodesArrayZoomed[d.col].nameWidthPx;
        DrawRect(-leftTextWidthR - PTREE_N2_GAP_PX, n2Dy * d.row, leftTextWidthR, n2Dy, RED_ARROW_COLOR); //highlight var name
        DrawRect(-leftTextWidthC - PTREE_N2_GAP_PX, n2Dy * d.col, leftTextWidthC, n2Dy, GREEN_ARROW_COLOR); //highlight var name
    }

    function MouseoverOnDiagN2(d) {
        //d=hovered element
        // console.log('MouseoverOnDiagN2:'); console.log(d);
        var hoveredIndexRC = d.col; //d.x == d.y == row == col
        var leftTextWidthHovered = d3RightTextNodesArrayZoomed[hoveredIndexRC].nameWidthPx;

        // Loop over all elements in the matrix looking for other cells in the same column as
        var lineWidth = Math.min(5, n2Dx * .5, n2Dy * .5);
        arrowMarker.attr("markerWidth", lineWidth * .4)
            .attr("markerHeight", lineWidth * .4);
        DrawRect(-leftTextWidthHovered - PTREE_N2_GAP_PX, n2Dy * hoveredIndexRC, leftTextWidthHovered, n2Dy, HIGHLIGHT_HOVERED_COLOR); //highlight hovered
        for (var i = 0; i < d3RightTextNodesArrayZoomed.length; ++i) {
            var leftTextWidthDependency = d3RightTextNodesArrayZoomed[i].nameWidthPx;
            var box = d3RightTextNodesArrayZoomedBoxInfo[i];
            if (matrix.node(hoveredIndexRC, i) !== undefined) { //i is column here
                if (i != hoveredIndexRC) {
                    new N2Arrow({
                        end: { col: i, row: i },
                        start: { col: hoveredIndexRC, row: hoveredIndexRC },
                        color: GREEN_ARROW_COLOR,
                        width: lineWidth
                    });
                    DrawRect(-leftTextWidthDependency - PTREE_N2_GAP_PX, n2Dy * i, leftTextWidthDependency, n2Dy, GREEN_ARROW_COLOR); //highlight var name
                }
            }

            if (matrix.node(i, hoveredIndexRC) !== undefined) { //i is row here
                if (i != hoveredIndexRC) {
                    new N2Arrow({
                        start: { col: i, row: i },
                        end: { col: hoveredIndexRC, row: hoveredIndexRC },
                        color: RED_ARROW_COLOR,
                        width: lineWidth
                    });
                    DrawRect(-leftTextWidthDependency - PTREE_N2_GAP_PX, n2Dy * i, leftTextWidthDependency, n2Dy, RED_ARROW_COLOR); //highlight var name
                }
            }
        }
    }

    function MouseoutN2() {
        n2Group.selectAll(".n2_hover_elements").remove();
    }

    function MouseClickN2(d) {
        var newClassName = "n2_hover_elements_" + d.row + "_" + d.col;
        var selection = n2Group.selectAll("." + newClassName);
        if (selection.size() > 0) {
            selection.remove();
        }
        else {
            n2Group.selectAll("path.n2_hover_elements, circle.n2_hover_elements")
                .attr("class", newClassName);
        }
    }

    function ReturnToRootButtonClick() {
        backButtonHistory.push({ "el": zoomedElement });
        forwardButtonHistory = [];
        SetupLeftClick(root);
        Update();
    }

    function UpOneLevelButtonClick() {
        if (zoomedElement === root) return;
        backButtonHistory.push({ "el": zoomedElement });
        forwardButtonHistory = [];
        SetupLeftClick(zoomedElement.parent);
        Update();
    }

    function CollapseOutputsButtonClick(startNode) {
        function CollapseOutputs(d) {
            if (d.subsystem_type && d.subsystem_type === "component") {
                d.isMinimized = true;
            }
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    CollapseOutputs(d.children[i]);
                }
            }
        }
        FindRootOfChangeFunction = FindRootOfChangeForCollapseUncollapseOutputs;
        TRANSITION_DURATION = TRANSITION_DURATION_SLOW;
        lastClickWasLeft = false;
        CollapseOutputs(startNode);
        Update();
    }

    function UncollapseButtonClick(startNode) {
        function Uncollapse(d) {
            if (d.type !== "param" && d.type !== "unconnected_param") {
                d.isMinimized = false;
            }
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    Uncollapse(d.children[i]);
                }
            }
        }
        FindRootOfChangeFunction = FindRootOfChangeForCollapseUncollapseOutputs;
        TRANSITION_DURATION = TRANSITION_DURATION_SLOW;
        lastClickWasLeft = false;
        Uncollapse(startNode);
        Update();
    }

    function CollapseToDepthSelectChange(newChosenCollapseDepth) {
        function CollapseToDepth(d, depth) {
            if (d.type === "param" || d.type === "unknown" || d.type === "unconnected_param") {
                return;
            }
            if (d.depth < depth) {
                d.isMinimized = false;
            }
            else {
                d.isMinimized = true;
            }
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    CollapseToDepth(d.children[i], depth);
                }
            }
        }

        chosenCollapseDepth = newChosenCollapseDepth;
        if (chosenCollapseDepth > zoomedElement.depth) {
            CollapseToDepth(root, chosenCollapseDepth);
        }
        FindRootOfChangeFunction = FindRootOfChangeForCollapseDepth;
        TRANSITION_DURATION = TRANSITION_DURATION_SLOW;
        lastClickWasLeft = false;
        Update();
    }

    function FontSizeSelectChange(fontSize) {
        for (var i = 8; i <= 14; ++i) {
            var newText = (i == fontSize) ? ("<b>" + i + "px</b>") : (i + "px");
            parentDiv.querySelector("#idFontSize" + i + "px").innerHTML = newText;
        }
        N2SVGLayout.fontSizePx = fontSize;
        TRANSITION_DURATION = TRANSITION_DURATION_FAST;
        UpdateSvgCss(svgStyleElement, fontSize);
        Update();
    }

    function VerticalResize(height) {
        for (var i = 600; i <= 1000; i += 50) {
            var newText = (i == height) ? ("<b>" + i + "px</b>") : (i + "px");
            parentDiv.querySelector("#idVerticalResize" + i + "px").innerHTML = newText;
        }
        for (var i = 2000; i <= 4000; i += 1000) {
            var newText = (i == height) ? ("<b>" + i + "px</b>") : (i + "px");
            parentDiv.querySelector("#idVerticalResize" + i + "px").innerHTML = newText;
        }
        ClearArrowsAndConnects();
        N2SVGLayout.heightPx = height;
        HEIGHT_PX = height;
        LEVEL_OF_DETAIL_THRESHOLD = height / 3;
        WIDTH_N2_PX = height;
        TRANSITION_DURATION = TRANSITION_DURATION_FAST;
        UpdateSvgCss(svgStyleElement, N2SVGLayout.fontSizePx);
        Update();
    }

    function ToggleSolverNamesCheckboxChange() {
        N2SVGLayout.toggleSolverNameType();
        // showLinearSolverNames = !showLinearSolverNames;
        parentDiv.querySelector("#toggleSolverNamesButtonId").className = !N2SVGLayout.showLinearSolverNames ? "myButton myButtonToggledOn" : "myButton";
        SetupLegend(d3, d3ContentDiv);
        Update();
    }

    function ShowPathCheckboxChange() {
        showPath = !showPath;
        parentDiv.querySelector("#currentPathId").style.display = showPath ? "block" : "none";
        parentDiv.querySelector("#showCurrentPathButtonId").className = showPath ? "myButton myButtonToggledOn" : "myButton";
    }

    function ToggleLegend() {
        showLegend = !showLegend;
        parentDiv.querySelector("#showLegendButtonId").className = showLegend ? "myButton myButtonToggledOn" : "myButton";
        SetupLegend(d3, d3ContentDiv);
    }

    function CreateDomLayout() {
        document.getElementById("searchButtonId").onclick = SearchButtonClicked;
    }

    function CreateToolbar() {
        var div = document.getElementById("toolbarDiv")
        div.querySelector("#returnToRootButtonId").onclick = ReturnToRootButtonClick;
        div.querySelector("#backButtonId").onclick = BackButtonPressed;
        div.querySelector("#forwardButtonId").onclick = ForwardButtonPressed;
        div.querySelector("#upOneLevelButtonId").onclick = UpOneLevelButtonClick;
        div.querySelector("#uncollapseInViewButtonId").onclick = function () { UncollapseButtonClick(zoomedElement); };
        div.querySelector("#uncollapseAllButtonId").onclick = function () { UncollapseButtonClick(root); };
        div.querySelector("#collapseInViewButtonId").onclick = function () { CollapseOutputsButtonClick(zoomedElement); };
        div.querySelector("#collapseAllButtonId").onclick = function () { CollapseOutputsButtonClick(root); };
        div.querySelector("#clearArrowsAndConnectsButtonId").onclick = ClearArrowsAndConnects;
        div.querySelector("#showCurrentPathButtonId").onclick = ShowPathCheckboxChange;
        div.querySelector("#showLegendButtonId").onclick = ToggleLegend;

        div.querySelector("#toggleSolverNamesButtonId").onclick = ToggleSolverNamesCheckboxChange;

        for (var i = 8; i <= 14; ++i) {
            var f = function (idx) {
                return function () { FontSizeSelectChange(idx); };
            }(i);
            div.querySelector("#idFontSize" + i + "px").onclick = f;
        }

        for (var i = 600; i <= 1000; i += 50) {
            var f = function (idx) {
                return function () { VerticalResize(idx); };
            }(i);
            div.querySelector("#idVerticalResize" + i + "px").onclick = f;
        }
        for (var i = 2000; i <= 4000; i += 1000) {
            var f = function (idx) {
                return function () { VerticalResize(idx); };
            }(i);
            div.querySelector("#idVerticalResize" + i + "px").onclick = f;
        }

        div.querySelector("#saveSvgButtonId").onclick = function () { SaveSvg(parentDiv) };
        div.querySelector("#helpButtonId").onclick = DisplayModal;
    }

    return {
        GetFontSize: function () { return N2SVGLayout.fontSizePx; },
        ResizeHeight: function (h) { VerticalResize(h); },
        Redraw: function () { Update(); }
    };
}

var zoomedElement = modelData.tree;
var updateFunc;
var mouseOverOffDiagN2;
var mouseOverOnDiagN2;
var mouseOutN2;
var mouseClickN2;
var treeData, connectionList;

var app = PtN2Diagram(document.getElementById("ptN2ContentDivId"), modelData);