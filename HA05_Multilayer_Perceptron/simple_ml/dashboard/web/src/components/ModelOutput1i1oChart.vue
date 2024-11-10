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
        model_output: 'drawCurve',
    },
    mounted() {
        this.$options.sockets.onopen = () => {
            this.$socket.send(JSON.stringify({
                type: 'poll',
                prop_names: ['train_dataset', 'test_dataset']
            }));
        }

        this.drawCurve();
    },
    methods: {
        isModelOutputValid(data) {
            return (
                data &&
                data.type === '1i1o' &&
                data.in &&
                Array.isArray(data.in) &&
                data.in.length === 1 &&
                data.in[0].range &&
                data.out &&
                Array.isArray(data.out) &&
                data.out.length === 1 &&
                data.out[0].range &&
                data.data &&
                Array.isArray(data.data)
            );
        },
        drawCurve() {
            const svg = d3.select(this.$refs.chart);
            svg.selectAll(".curve").remove();
            svg.selectAll(".train_scatter").remove();
            svg.selectAll(".test_scatter").remove();

            // model_output data
            let mo = this.model_output;
            let mo_flag = this.isModelOutputValid(mo);
            if (!mo_flag) {
                mo = {
                    type: '1i1o',
                    in: [
                        {name: 'Input', range: [0, 1, 1]},
                    ],
                    out: [
                        {name: 'Output', range: [-1, 1]}
                    ],
                    data: [0]
                };
            }
            const x_range = mo.in[0].range;
            const output_range = mo.out[0].range;
            const data = mo.data;
            const featureName = (mo.in[0].name !== undefined) ? mo.in[0].name : "Input";
            const outputName = (mo.out[0].name !== undefined) ? mo.out[0].name : "Output";

            // _dataset data
            let dstr = this.train_dataset, dste = this.test_dataset;
            if (!this.isDatasetValid(dstr)) {
                dstr = {
                    type: '1i1o',
                    data_in: [],
                    data_out: []
                }
            }
            if (!this.isDatasetValid(dste)) {
                dste = {
                    type: '1i1o',
                    data_in: [],
                    data_out: []
                }
            }

            // scale
            const x = d3.scaleLinear()
                .domain([x_range[0], x_range[1]])
                .range([60, this.width - 50]);

            const y = d3.scaleLinear()
                .domain([output_range[0], output_range[1]])
                .range([this.height - 45, 20]);

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

            // line generator
            const line = d3.line()
                .x((d, i) => x(x_range[0] + i * (x_range[1] - x_range[0]) / x_range[2]))
                .y(d => y(d));

            // scatters, TODO: filter outside points
            const plotPoints = (dataset, cls, borderColor) => {
                dataset.data_in.forEach((d, i) => {
                    svg.append("circle")
                        .attr("class", cls)
                        .attr("cx", x(d))
                        .attr("cy", y(dataset.data_out[i]))
                        .attr("r", this.dotRadius)
                        .attr("fill", "#f59322")
                        .attr("stroke", borderColor)
                        .attr("stroke-width", this.dotBorder);
                });
            };
            if (mo_flag) {
                plotPoints(dstr, 'train_scatter', "#ffffff");
                plotPoints(dste, 'test_scatter', "#000000");
            }

            // curve
            svg.append("path")
                .data([data])
                .attr("class", "curve")
                .attr("fill", "none")
                .attr("stroke", "#000000")
                .attr("stroke-width", 2)
                .attr("d", line);

            // dynamic x ticks
            const xTicks = Math.min(data.length / 5, 10);
            const xAxis = d3.axisBottom(x).ticks(xTicks);
            svg.append("g")
                .attr("class", "curve")
                .attr("transform", `translate(0, ${this.height - 45})`)
                .call(xAxis);

            // x label
            svg.append("text")
                .attr("class", "curve")
                .attr("x", this.width / 2)
                .attr("y", this.height - 10)
                .style("text-anchor", "middle")
                .text(featureName);

            // dynamic y ticks
            const yTicks = 5;
            const yAxis = d3.axisLeft(y).ticks(yTicks);
            svg.append("g")
                .attr("class", "curve")
                .attr("transform", "translate(60, 0)")
                .call(yAxis);

            // y label
            svg.append("text")
                .attr("class", "curve")
                .attr("transform", "rotate(-90)")
                .attr("y", 20)
                .attr("x", -this.height / 2)
                .style("text-anchor", "middle")
                .text(outputName);
        },
        isDatasetValid(data) {
            return (
                data &&
                data.type === '1i1o' &&
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