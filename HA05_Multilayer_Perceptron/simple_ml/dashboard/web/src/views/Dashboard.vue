<template>
    <div class="dashboard">
        <EpochIndicator :epoch="epoch" />
        <ControlButtons @reset="handleReset" />
        <ValueWithRespectToEpochChart
            :values="train_loss_buffer"
            :width="600"
            :height="200"
            yAxisLabel="Train Loss"
        />
        <ValueWithRespectToEpochChart
            :values="train_accuracy_buffer"
            :width="600"
            :height="200"
            yAxisLabel="Train Accuracy"
        />
    </div>
</template>

<script>
import EpochIndicator from "@/components/EpochIndicator.vue";
import ControlButtons from "@/components/ControlButtons.vue";
import ValueWithRespectToEpochChart from "@/components/ValueWithRespectToEpochChart.vue";

export default {
    components: {
        EpochIndicator,
        ControlButtons,
        ValueWithRespectToEpochChart
    },
    data() {
        return {
            epoch: 0,
            train_loss_buffer: [],
            train_accuracy_buffer: []
            // TODO
        }
    },
    mounted() {
        this.$options.sockets.onmessage = (msg) => {
            const data = JSON.parse(msg.data);
            // TODO: type validation
            if (data.epoch !== undefined) {
                this.epoch = data.epoch;
            }
            if (data.train_loss_buffer !== undefined) {
                this.train_loss_buffer = data.train_loss_buffer;
            }
            if (data.train_accuracy_buffer !== undefined) {
                this.train_accuracy_buffer = data.train_accuracy_buffer;
            }
            // TODO
        }
    },
    methods: {
        handleReset() {
            this.epoch = 0;
            this.train_loss_buffer = [];
            this.train_accuracy_buffer = [];
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