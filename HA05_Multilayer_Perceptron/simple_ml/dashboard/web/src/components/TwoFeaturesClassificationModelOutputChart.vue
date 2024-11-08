<template>
    <div ref="chart" class="chart"></div>
</template>

<script>
import * as d3 from 'd3';

import global_config from "@/config.js";

export default {
    data() {
        return {
            data: [],
            modelOutput: []
        };
    },
    mounted() {
        this.initChart();
        // this.connectWebSocket(); // 任务开始后、刷新时再连接，提供 task_id 和 view_id。
    },
    methods: {
        initChart() {
            const svg = d3.select(this.$refs.chart)
                .append('svg')
                .attr('width', 600)
                .attr('height', 600);

            // this.xScale = d3.scaleLinear().domain([-10, 10]).range([0, 600]);
            // this.yScale = d3.scaleLinear().domain([-10, 10]).range([600, 0]);
            //
            // svg.append('g').attr('class', 'data-points');
            // svg.append('g').attr('class', 'model-output');
        },
        connectWebSocket() {
            const ws = new WebSocket((new URL("/notify", global_config.ws_address)).href);

            ws.onmessage = (event) => {
                // const { dataPoints, modelOutput } = JSON.parse(event.data);
                // this.data = dataPoints;
                // this.modelOutput = modelOutput;
                // this.updateChart();
            };
        },
        updateChart() {
            // const svg = d3.select(this.$refs.chart).select('svg');
            //
            // const dataPoints = svg.select('.data-points')
            //     .selectAll('circle')
            //     .data(this.data, d => `${d.x}-${d.y}`);
            //
            // dataPoints.enter()
            //     .append('circle')
            //     .attr('cx', d => this.xScale(d.x))
            //     .attr('cy', d => this.yScale(d.y))
            //     .attr('r', 3)
            //     .attr('fill', 'blue');
            //
            // const modelPoints = svg.select('.model-output')
            //     .selectAll('circle')
            //     .data(this.modelOutput, d => `${d.x}-${d.y}`);
            //
            // modelPoints.enter()
            //     .append('circle')
            //     .attr('cx', d => this.xScale(d.x))
            //     .attr('cy', d => this.yScale(d.y))
            //     .attr('r', 3)
            //     .attr('fill', 'red');
        }
    }
};
</script>

<style scoped>
.chart {
    width: 600px;
    height: 600px;
    border: 1px solid #ddd;
}
</style>