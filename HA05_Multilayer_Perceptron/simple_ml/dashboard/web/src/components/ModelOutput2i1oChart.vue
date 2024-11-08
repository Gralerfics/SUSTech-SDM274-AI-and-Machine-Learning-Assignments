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
        width: {
            type: Number,
            default: 500
        },
        height: {
            type: Number,
            default: 500
        }
    },
    watch: {
        model_output(newData) {
            if (this.isValidData(newData)) {
                this.drawChart(newData);
            } else {
                this.clearChart();
                console.warn("Invalid model_output data");
            }
        }
    },
    mounted() {
        if (this.isValidData(this.model_output)) {
            this.drawChart(this.model_output);
        } else {
            console.warn("Invalid model_output data on mount");
        }
    },
    methods: {
        isValidData(data) {
            return (
                data &&
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
        drawChart(data) {
            const svg = d3.select(this.$refs.chart);
            svg.selectAll("*").remove();

            const viewWidth = this.width - 80;
            const viewHeight = this.height - 80;

            const x1_range = data.in[0].range;
            const x2_range = data.in[1].range;

            const cellWidth = viewWidth / x1_range[2] + 1;
            const cellHeight = viewHeight / x2_range[2] + 1;

            const xScale = d3.scaleLinear()
                .domain([x1_range[0], x1_range[1]])
                .range([60, this.width - 20]);

            const yScale = d3.scaleLinear()
                .domain([x2_range[0], x2_range[1]])
                .range([this.height - 60, 20]);

            const output_range = data.out[0].range;
            const gridData = data.data;

            // colormap
            const npp_cn = "rgb(237, 153, 65)";
            const npp_cm = "rgb(233, 233, 233)";
            const npp_cp = "rgb(39, 122, 185)";
            const colorScale = d3.scaleLinear()
                .domain([output_range[0], 0, output_range[1]])
                .range([npp_cn, npp_cm, npp_cp]);

            // mesh grids
            gridData.forEach((row, i) => {
                row.forEach((value, j) => {
                    svg.append("rect")
                        .attr("x", xScale(x1_range[0] + j * (x1_range[1] - x1_range[0]) / x1_range[2]))
                        .attr("y", yScale(x2_range[0] + (i + 1) * (x2_range[1] - x2_range[0]) / x2_range[2]))
                        .attr("width", cellWidth)
                        .attr("height", cellHeight)
                        .attr("fill", colorScale(value));
                });
            });

            // axes
            const xAxis = d3.axisBottom(xScale).ticks(10);
            svg.append("g")
                .attr("transform", `translate(0, ${this.height - 60})`)
                .call(xAxis)

            const featureName1 = (data.in[0].name !== undefined) ? data.in[0].name : "Feature 1";
            svg.append("text")
                .attr("x", this.width / 2 + 20)
                .attr("y", this.height - 20)
                .style("text-anchor", "middle")
                .text(featureName1);

            const yAxis = d3.axisLeft(yScale).ticks(10);
            svg.append("g")
                .attr("transform", "translate(60, 0)")
                .call(yAxis)

            const featureName2 = (data.in[1].name !== undefined) ? data.in[1].name : "Feature 2";
            svg.append("text")
                .attr("x", -this.height / 2)
                .attr("y", 30)
                .attr("transform", "rotate(-90)")
                .style("text-anchor", "middle")
                .text(featureName2);
        },
        clearChart() {
            const svg = d3.select(this.$refs.chart);
            svg.selectAll("*").remove();
        }
    }
};
</script>

<style scoped>
div {
    border: 1px solid #ccc;
}
</style>