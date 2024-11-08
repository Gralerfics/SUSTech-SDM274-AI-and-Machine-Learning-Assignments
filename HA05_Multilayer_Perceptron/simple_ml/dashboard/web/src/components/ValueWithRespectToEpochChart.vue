<template>
    <svg ref="chart" :width="width" :height="height"></svg>
</template>

<script>
import * as d3 from 'd3';

export default {
    props: {
        values: {
            type: Array,
            required: true
        },
        width: {
            type: Number,
            default: 500
        },
        height: {
            type: Number,
            default: 300
        },
        yAxisDomain: {
            type: Array,
            default: [null, null]
        },
        yAxisLabel: {
            type: String,
            default: "Loss"
        }
    },
    watch: {
        values(newData) {
            this.drawChart(newData);
        }
    },
    mounted() {
        this.drawChart(this.values);
    },
    methods: {
        drawChart(data) {
            const svg = d3.select(this.$refs.chart);
            svg.selectAll("*").remove();

            const width = this.width;
            const height = this.height;

            // scale
            const x = d3.scaleLinear()
                .domain([0, data.length - 1])
                .range([60, width - 50]);

            const y = d3.scaleLinear()
                .domain([(this.yAxisDomain[0] === null) ? d3.min(data) : this.yAxisDomain[0], (this.yAxisDomain[1] === null) ? d3.max(data) : this.yAxisDomain[1]])
                .range([height - 45, 20]);

            // line generator
            const line = d3.line()
                .x((d, i) => x(i))
                .y(d => y(d));

            // draw curve
            svg.append("path")
                .data([data])
                .attr("fill", "none")
                .attr("stroke", "black")
                .attr("stroke-width", 2)
                .attr("d", line);

            // dynamic x ticks
            const xTicks = Math.min(data.length / 5, 10);
            const xAxis = d3.axisBottom(x).ticks(xTicks);
            svg.append("g")
                .attr("transform", `translate(0, ${height - 45})`)
                .call(xAxis);

            // x label
            svg.append("text")
                .attr("x", width / 2)
                .attr("y", height - 10)
                .style("text-anchor", "middle")
                .text("Epoch");

            // dynamic y ticks
            const yTicks = 5;
            const yAxis = d3.axisLeft(y).ticks(yTicks);
            svg.append("g")
                .attr("transform", "translate(60, 0)")
                .call(yAxis);

            // y label
            svg.append("text")
                .attr("transform", "rotate(-90)")
                .attr("y", 20)
                .attr("x", -height / 2)
                .style("text-anchor", "middle")
                .text(this.yAxisLabel);
        }
    }
};
</script>

<style scoped>
svg {
    border: 1px solid #ccc;
}
</style>
