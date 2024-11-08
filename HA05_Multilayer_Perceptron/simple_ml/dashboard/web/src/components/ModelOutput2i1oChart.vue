<template>
    <svg ref="chart" :width="width" :height="height"></svg>
</template>

<script>
import * as d3 from 'd3';

export default {
    props: {
        model_output: {
            type: Object,
            required: true
        },
        train_dataset: {
            type: Object,
            required: true
        },
        test_dataset: {
            type: Object,
            required: true
        },
        width: {
            type: Number,
            default: 500
        },
        height: {
            type: Number,
            default: 500
        },
        quantizedColorNumber: {
            type: Number,
            default: 50
        },
        dotRadius: {
            type: Number,
            default: 2.5
        },
        dotBorder: {
            type: Number,
            default: 0.5
        }
    },
    watch: {
        model_output: 'drawHeatmap',
    },
    mounted() {
        this.$options.sockets.onopen = () => {
            this.$socket.send(JSON.stringify({
                type: 'poll',
                prop_names: ['train_dataset', 'test_dataset']
            }));
        }

        this.drawHeatmap();
    },
    methods: {
        isModelOutputValid(data) {
            return (
                data &&
                data.type === '2i1o' &&
                data.in &&
                Array.isArray(data.in) &&
                data.in.length === 2 &&
                data.in[0].range &&
                data.in[1].range &&
                data.out &&
                Array.isArray(data.out) &&
                data.out.length === 1 &&
                data.out[0].range &&
                data.data &&
                Array.isArray(data.data) &&
                Array.isArray(data.data[0])
            );
        },
        drawHeatmap() {
            const svg = d3.select(this.$refs.chart);
            svg.selectAll(".heatmap").remove();
            svg.selectAll(".train_scatter").remove();
            svg.selectAll(".test_scatter").remove();

            // model_output data
            let mo = this.model_output;
            let mo_flag = this.isModelOutputValid(mo);
            if (!mo_flag) {
                mo = {
                    type: '2i1o',
                    in: [
                        {name: 'Feature 1', range: [0, 1, 1]},
                        {name: 'Feature 2', range: [0, 1, 1]}
                    ],
                    out: [
                        {name: 'Output', range: [-1, 1]}
                    ],
                    data: [[0]]
                };
            }
            const x1_range = mo.in[0].range;
            const x2_range = mo.in[1].range;
            const output_range = mo.out[0].range;
            const gridData = mo.data;
            const featureName1 = (mo.in[0].name !== undefined) ? mo.in[0].name : "Feature 1";
            const featureName2 = (mo.in[1].name !== undefined) ? mo.in[1].name : "Feature 2";

            // _dataset data
            let dstr = this.train_dataset, dste = this.test_dataset;
            if (!this.isDatasetValid(dstr)) {
                dstr = {
                    type: '2i1o',
                    data_in: [],
                    data_out: []
                }
            }
            if (!this.isDatasetValid(dste)) {
                dste = {
                    type: '2i1o',
                    data_in: [],
                    data_out: []
                }
            }

            // scale
            const xScale = d3.scaleLinear()
                .domain([x1_range[0], x1_range[1]])
                .range([60, this.width - 20]);

            const yScale = d3.scaleLinear()
                .domain([x2_range[0], x2_range[1]])
                .range([this.height - 60, 20]);

            // colormap
            let tmpScale = d3.scaleLinear()
                .domain([0, .5, 1])
                .range(["#f59322", "#e8eaeb", "#0877bd"])
                .clamp(true);
            let colors = d3.range(0, 1 + 1e-9, 1 / this.quantizedColorNumber).map(a => {
                return tmpScale(a);
            });
            const colorScale = d3.scaleQuantize()
                .domain([output_range[0], output_range[1]])
                .range(colors);

            // mesh grids
            const viewWidth = this.width - 80;
            const viewHeight = this.height - 80;

            const cellWidth = viewWidth / x1_range[2] + 1;
            const cellHeight = viewHeight / x2_range[2] + 1;

            gridData.forEach((row, i) => {
                row.forEach((value, j) => {
                    svg.append("rect")
                        .attr("class", "heatmap")
                        .attr("x", xScale(x1_range[0] + j * (x1_range[1] - x1_range[0]) / x1_range[2]))
                        .attr("y", yScale(x2_range[0] + (i + 1) * (x2_range[1] - x2_range[0]) / x2_range[2]))
                        .attr("width", cellWidth)
                        .attr("height", cellHeight)
                        .attr("fill", colorScale(value));
                });
            });

            // scatters
            const plotPoints = (dataset, cls, borderColor) => {
                dataset.data_in.forEach((point, index) => {
                    svg.append("circle")
                        .attr("class", cls)
                        .attr("cx", xScale(point[0]))
                        .attr("cy", yScale(point[1]))
                        .attr("r", this.dotRadius)
                        .attr("fill", colorScale(dataset.data_out[index]))
                        .attr("stroke", borderColor)
                        .attr("stroke-width", this.dotBorder);
                });
            };
            if (mo_flag) {
                plotPoints(dstr, 'train_scatter', "#ffffff");
                plotPoints(dste, 'test_scatter', "#000000");
            }

            // axes
            const xAxis = d3.axisBottom(xScale).ticks(10);
            svg.append("g")
                .attr("class", "heatmap")
                .attr("transform", `translate(0, ${this.height - 60})`)
                .call(xAxis)

            svg.append("text")
                .attr("class", "heatmap")
                .attr("x", this.width / 2 + 20)
                .attr("y", this.height - 20)
                .style("text-anchor", "middle")
                .text(featureName1);

            const yAxis = d3.axisLeft(yScale).ticks(10);
            svg.append("g")
                .attr("class", "heatmap")
                .attr("transform", "translate(60, 0)")
                .call(yAxis)

            svg.append("text")
                .attr("class", "heatmap")
                .attr("x", -this.height / 2 + 20)
                .attr("y", 30)
                .attr("transform", "rotate(-90)")
                .style("text-anchor", "middle")
                .text(featureName2);
        },
        isDatasetValid(data) {
            return (
                data &&
                data.type === '2i1o' &&
                data.data_in &&
                Array.isArray(data.data_in) &&
                data.data_out &&
                Array.isArray(data.data_out)
            );
        }
    }
};
</script>

<style scoped>
svg {
    border: 1px solid #ccc;
}
</style>