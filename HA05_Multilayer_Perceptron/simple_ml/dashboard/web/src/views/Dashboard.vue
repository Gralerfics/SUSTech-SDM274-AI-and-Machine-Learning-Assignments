<template>
    <div class="dashboard">
        <EpochIndicator
            :epoch="viewDatabase.epoch"
        />

        <ControlButtons />

        <ModelOutput2i1oChart
            :model_output="viewDatabase.model_output"
            :width="500"
            :height="500"
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
            viewDatabase: {} // 各组件应有 isValidData 和 drawChart 方法，若数据不合法（null/undefined 等）则应绘制空白图（例如以赋特定值的方式）
        }
    },
    mounted() {
        this.$options.sockets.onmessage = (msg) => {
            const data = JSON.parse(msg.data);
            if (data.type === 'poll_response') {
                if (data.prop_name !== undefined && data.prop_value !== undefined) {
                    this.updateViewDatabaseEntry(data.prop_name, data.prop_value);
                    // this.viewDatabase.model_output = data.prop_value;
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
                this.$socket.send(JSON.stringify({ // TODO: sendObj
                    type: 'poll',
                    prop_name: ''
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
        recursivelyMerge(target, source) {
            for (const key in source) {
                if (
                    key in target &&
                    source[key].constructor === Object &&
                    target[key].constructor === Object
                ) {
                    this.recursivelyMerge(target[key], source[key]);
                } else {
                    target[key] = source[key];
                }
            }
            return target;
        },
        getViewDatabaseEntry(propName) {
            const keys = propName.split(".").slice(1);
            let current = this.viewDatabase;

            for (const key of keys) {
                if (current[key] === undefined) {
                    return undefined;
                }
                current = current[key];
            }

            return current;
        },
        updateViewDatabaseEntry(propName, propValue) {
            const path = propName.split('.').slice(1);

            let current = this.viewDatabase;

            for (let i = 0; i < path.length; i ++) {
                const key = path[i];

                if (i === path.length - 1 && propValue.constructor !== Object) {
                    current[key] = propValue;
                } else {
                    if (!current[key]) {
                        current[key] = {};
                    }
                    current = current[key];
                }
            }

            if (propValue.constructor === Object) {
                this.recursivelyMerge(current, propValue);
            }
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