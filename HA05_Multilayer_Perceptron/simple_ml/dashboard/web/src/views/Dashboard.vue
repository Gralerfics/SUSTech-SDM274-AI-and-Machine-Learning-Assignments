<template>
    <div class="dashboard">
        <EpochIndicator :epoch="epoch" />

        <ControlButtons @reset="handleReset" />

        <ModelOutput2i1oChart
            :model_output="model_output"
            :width="500"
            :height="500"
        />

        <ValueWithRespectToEpochChart
            :values="train_loss_buffer"
            :width="600"
            :height="200"
            :yAxisDomain="[0, null]"
            yAxisLabel="Train Loss"
        />

        <ValueWithRespectToEpochChart
            :values="train_accuracy_buffer"
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
            epoch: 0,
            train_loss_buffer: [],
            train_accuracy_buffer: [],
            model_output: {}
            // TODO
        }
    },
    mounted() {
        this.$options.sockets.onmessage = (msg) => {
            const data = JSON.parse(msg.data);
            // TODO: type validation
            if (data.epoch !== undefined && data.model_output !== null) {
                this.epoch = data.epoch;
            }
            if (data.train_loss_buffer !== undefined && data.model_output !== null) {
                this.train_loss_buffer = data.train_loss_buffer;
            }
            if (data.train_accuracy_buffer !== undefined && data.model_output !== null) {
                this.train_accuracy_buffer = data.train_accuracy_buffer;
            }
            if (data.model_output !== undefined && data.model_output !== null) {
                this.model_output = data.model_output;
            }
            // TODO
        }
    },
    methods: {
        handleReset() {
            this.epoch = 0;
            this.train_loss_buffer = [];
            this.train_accuracy_buffer = [];
            this.model_output = {};
            // TODO
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