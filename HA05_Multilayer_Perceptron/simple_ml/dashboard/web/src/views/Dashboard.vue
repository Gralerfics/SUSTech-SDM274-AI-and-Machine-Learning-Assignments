<template>
    <div class="dashboard">
        <EpochIndicator
            :epoch="viewDatabase.epoch"
        />

        <ControlButtons
            :taskConfig="taskConfig"
            @refreshTask="refreshTask"
        />

        <ModelOutput2i1oChart
            :model_output="viewDatabase.model_output"
            :train_dataset="viewDatabase.train_dataset"
            :test_dataset="viewDatabase.test_dataset"
            :width="300"
            :height="300"
        />

        <ValueWithRespectToEpochChart
            :values="viewDatabase.train_loss_buffer"
            :width="600"
            :height="200"
            :yAxisDomain="[0, null]"
            yAxisLabel="Train Loss"
        />

        <ValueWithRespectToEpochChart
            :values="viewDatabase.train_accuracy_buffer"
            :width="600"
            :height="200"
            :yAxisDomain="[0, 1]"
            yAxisLabel="Train Accuracy"
        />
    </div>
</template>

<script>
import EpochIndicator from "@/components/EpochIndicator.vue";
import ControlButtons from "@/components/ControlButtons.vue";
import ValueWithRespectToEpochChart from "@/components/ValueWithRespectToEpochChart.vue";
import ModelOutput2i1oChart from "@/components/ModelOutput2i1oChart.vue";

export default {
    components: {
        EpochIndicator,
        ControlButtons,
        ValueWithRespectToEpochChart,
        ModelOutput2i1oChart
    },
    data() {
        return {
            isWebsocketConnected: false,
            taskConfig: {
                dataset: {
                    type: 'builtin',
                    name: 'generate_2d_classification_circle',
                    params: {
                        N: 500
                    },
                    proc: 'label_split_for_2d_classification_dataset',
                    test_ratio: 0.2,
                    batch_size: 10
                },
                model: [
                    { type: 'Linear', params: { in_dim: 2, out_dim: 4 } },
                    { type: 'Sigmoid' },
                    { type: 'Linear', params: { in_dim: 4, out_dim: 2 } },
                    { type: 'ReLU' },
                    { type: 'Linear', params: { in_dim: 2, out_dim: 1 } }
                ],
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
            },
            viewDatabase: {} // 各组件若数据不合法（null/undefined 等）应绘制空白图（例如以赋特定值的方式）
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
                    prop_names: ['epoch', 'model_output', 'train_loss_buffer', 'train_accuracy_buffer'],
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
        }
    }
};
</script>

<style>
.dashboard {
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 20px;
    align-items: center;
}
</style>