<template>
    <div class="dashboard">
        <div class="left-column">
            <div class="indicator-bar">
                <EpochIndicator
                    :epoch="viewDatabase.epoch"
                />

                <ControlButtons
                    :taskConfig="taskConfig"
                    @refreshTask="refreshTask"
                />
            </div>

            <div class="params-panel">
                <div style="color: #555; font-weight: bold">
                    Dataset
                    <hr />
                </div>
                <div class="param-field">
                    <label>func</label>
<!--                    <input type="text" v-model="taskConfig.dataset.name" />-->
                    <select v-model="taskConfig.dataset.name">
                        <option value="generate_2d_classification_circle">2D Classification (Circle)</option>
                        <option value="generate_2d_classification_exclusive_or">2D Classification (XOR)</option>
                        <option value="generate_2d_classification_example_scatter">2D Classification (Scatter)</option>
                        <option value="generate_1d_regression_with_function">1D Regression (Non-linear function)</option>
                    </select>
                </div>
                <div class="param-field">
                    <label>proc</label>
<!--                    <input type="text" v-model="taskConfig.dataset.proc" />-->
                    <select v-model="taskConfig.dataset.proc">
                        <option value="label_split_for_single_output_dataset">Single Output Splitter</option>
                    </select>
                </div>
                <div class="param-field">
                    <label>N</label>
                    <input type="number" v-model="taskConfig.dataset.params.N" min="0" />
                </div>
                <div class="param-field">
                    <label>noise</label>
                    <input type="number" v-model="taskConfig.dataset.params.noise" min="0" />
                </div>
                <div class="param-field">
                    <label>test_ratio</label>
                    <input type="number" v-model="taskConfig.dataset.test_ratio" min="0" />
                </div>
            </div>

            <div class="params-panel">
                <div style="color: #555; font-weight: bold">
                    Training
                    <hr />
                </div>
                <div class="param-field">
                    <label>optimizer</label>
<!--                    <input type="text" v-model="taskConfig.optimizer.type" />-->
                    <select v-model="taskConfig.optimizer.type">
                        <option value="GD">Gradient Descent</option>
                        <option value="Adam">Adam</option>
                    </select>
                </div>
                <div class="param-field">
                    <label>lr</label>
                    <input type="number" v-model="taskConfig.optimizer.params.lr" min="0" />
                </div>
                <div class="param-field">
                    <label>batch_size</label>
                    <input type="number" v-model="taskConfig.dataset.batch_size" min="0" />
                </div>
            </div>

            <div class="params-panel">
                <div style="color: #555; font-weight: bold">
                    Charts Visibility
                    <hr />
                </div>
                <div v-for="(value, key) in chartsVisible" :key="key">
                    <div class="param-field">
                        <label>{{ key }}</label>
                        <input type="checkbox" v-model="chartsVisible[key]" />
                    </div>
                </div>
            </div>

            <ModelBuilder
                :modelBlocks="taskConfig.model"
                @update:modelBlocks="updateModelBlocks"
            />
        </div>
        <div class="right-column">
            <ModelOutput2i1oChart
                v-if="chartsVisible.model_output && viewDatabase.model_output && viewDatabase.model_output.type === '2i1o'"
                :model_output="viewDatabase.model_output"
                :train_dataset="viewDatabase.train_dataset"
                :test_dataset="viewDatabase.test_dataset"
                :width="300"
                :height="300"
            />

            <ModelOutput1i1oChart
                v-if="chartsVisible.model_output && viewDatabase.model_output && viewDatabase.model_output.type === '1i1o'"
                :model_output="viewDatabase.model_output"
                :train_dataset="viewDatabase.train_dataset"
                :test_dataset="viewDatabase.test_dataset"
                :width="600"
                :height="400"
            />

            <ValueWithRespectToEpochChart
                v-if="chartsVisible.train_loss_buffer"
                :values="viewDatabase.train_loss_buffer"
                :width="600"
                :height="200"
                :yAxisDomain="[0, null]"
                yAxisLabel="Train Loss"
            />

            <ValueWithRespectToEpochChart
                v-if="chartsVisible.train_accuracy_buffer"
                :values="viewDatabase.train_accuracy_buffer"
                :width="600"
                :height="200"
                :yAxisDomain="[0, 1]"
                yAxisLabel="Train Accuracy"
            />

            <ValueWithRespectToEpochChart
                v-if="chartsVisible.train_r2_buffer"
                :values="viewDatabase.train_r2_buffer"
                :width="600"
                :height="200"
                :yAxisDomain="[null, 1]"
                yAxisLabel="Train R2"
            />

            <ValueWithRespectToEpochChart
                v-if="chartsVisible.test_loss_buffer"
                :values="viewDatabase.test_loss_buffer"
                :width="600"
                :height="200"
                :yAxisDomain="[0, null]"
                yAxisLabel="Test Loss"
            />

            <ValueWithRespectToEpochChart
                v-if="chartsVisible.test_accuracy_buffer"
                :values="viewDatabase.test_accuracy_buffer"
                :width="600"
                :height="200"
                :yAxisDomain="[0, 1]"
                yAxisLabel="Test Accuracy"
            />

            <ValueWithRespectToEpochChart
                v-if="chartsVisible.test_r2_buffer"
                :values="viewDatabase.test_r2_buffer"
                :width="600"
                :height="200"
                :yAxisDomain="[null, 1]"
                yAxisLabel="Test R2"
            />
        </div>
    </div>
</template>

<script>
import EpochIndicator from "@/components/EpochIndicator.vue";
import ControlButtons from "@/components/ControlButtons.vue";
import ValueWithRespectToEpochChart from "@/components/ValueWithRespectToEpochChart.vue";
import ModelOutput1i1oChart from "@/components/ModelOutput1i1oChart.vue";
import ModelOutput2i1oChart from "@/components/ModelOutput2i1oChart.vue";

import ModelBuilder from '@/components/ModelBuilder.vue'

export default {
    components: {
        EpochIndicator,
        ControlButtons,
        ValueWithRespectToEpochChart,
        ModelOutput1i1oChart,
        ModelOutput2i1oChart,
        ModelBuilder
    },
    data() {
        return {
            isWebsocketConnected: false,
            taskConfig: {
                // dataset: {
                //     type: 'builtin',
                //     name: 'generate_2d_classification_circle',
                //     params: {
                //         N: 500
                //     },
                //     proc: 'label_split_for_single_output_dataset',
                //     test_ratio: 0.2,
                //     batch_size: 10
                // },
                // model: [
                //     { type: 'Linear', params: { in_dim: 2, out_dim: 4 } },
                //     { type: 'Sigmoid' },
                //     { type: 'Linear', params: { in_dim: 4, out_dim: 2 } },
                //     { type: 'ReLU' },
                //     { type: 'Linear', params: { in_dim: 2, out_dim: 1 } }
                // ],
                model: [
                    { type: 'Linear', params: { in_dim: 1, out_dim: 7 } },
                    { type: 'Sigmoid' },
                    { type: 'Linear', params: { in_dim: 7, out_dim: 13 } },
                    { type: 'Sigmoid' },
                    { type: 'Linear', params: { in_dim: 13, out_dim: 1 } }
                ],
                dataset: {
                    type: 'builtin',
                    name: 'generate_1d_regression_with_function',
                    params: {
                        N: 1000,
                        noise: 0.2
                    },
                    proc: 'label_split_for_single_output_dataset',
                    test_ratio: 0.2,
                    batch_size: 10
                },
                loss: {
                    type: 'MSELoss',
                    params: {}
                },
                optimizer: {
                    type: 'GD',
                    params: {
                        lr: 0.01
                    }
                }
                // optimizer: {
                //     type: 'Adam',
                //     params: {
                //         lr: 0.01
                //     }
                // }
            },
            // propAcquires: ['model_output', 'train_loss_buffer', 'train_accuracy_buffer', 'test_loss_buffer', 'test_accuracy_buffer'],
            // propAcquires: ['model_output', 'train_loss_buffer', 'train_r2_buffer', 'test_loss_buffer', 'test_r2_buffer'],
            propAcquires: ['model_output', 'train_loss_buffer', 'train_accuracy_buffer', 'test_loss_buffer', 'test_accuracy_buffer', 'train_loss_buffer', 'train_r2_buffer', 'test_loss_buffer', 'test_r2_buffer'],
            chartsVisible: {
                model_output: true,
                train_loss_buffer: true,
                train_accuracy_buffer: true,
                train_r2_buffer: true,
                test_loss_buffer: true,
                test_accuracy_buffer: true,
                test_r2_buffer: true
            },
            viewDatabase: {} // 各组件若数据不合法 (null/undefined 等) 应绘制空白图，例如以赋特定值的方式
        }
    },
    mounted() {
        this.$options.sockets.onmessage = (msg) => {
            const data = JSON.parse(msg.data);
            if (data.type === 'poll_response') {
                if (data.prop_values !== undefined) {
                    Object.assign(this.viewDatabase, data.prop_values);
                }
            }
        }

        this.$options.sockets.onopen = () => {
            this.isWebsocketConnected = true;
        }

        this.$options.sockets.onclose = () => {
            this.isWebsocketConnected = false;
        }

        this.pollInterval = setInterval(() => {
            if (this.isWebsocketConnected) {
                this.$socket.send(JSON.stringify({
                    type: 'poll',
                    prop_names: ['epoch'].concat(this.propAcquires),
                }));
            }
        }, 1000 / 10);
    },
    beforeUnmount() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }
    },
    methods: {
        refreshTask() {
            this.$socket.send(JSON.stringify({
                type: 'poll',
                prop_names: ['train_dataset', 'test_dataset']
            }));
        },
        updateModelBlocks(newModelBlocks) {
            this.taskConfig.model = newModelBlocks
        }
    }
};
</script>

<style>
.dashboard {
    display: flex;
    justify-content: space-around;
    gap: 20px;
    padding: 20px;
}

.left-column, .right-column {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.left-column {
    width: 40%;
}

.right-column {
    width: 50%;
}

.indicator-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 10px;
}

.params-panel {
    background-color: #f0f0f0;
    padding: 10px;
    border: 1px solid #ccc;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.params-panel .param-field {
    display: flex;
    align-items: center;
}

.params-panel .param-field label {
    flex: 1;
    font-style: italic;
    color: #555;
}

.params-panel .param-field input[type="number"], input[type="text"] {
    flex: 3;
    padding: 5px;
    border: none;
    border-radius: 8px;
    background-color: #fff;
    box-shadow: 0px 1px 3px rgba(0, 0, 0, 0.1);
    text-align: right;
}

.params-panel .param-field select {
    flex: 3;
    padding: 4px;
    border: none;
    border-radius: 8px;
    background-color: #fff;
    box-shadow: 0px 1px 3px rgba(0, 0, 0, 0.1);
    text-align: right;
    padding-right: 24px;
}
</style>