<template>
    <div class="dashboard">
        <EpochIndicator :epoch="epoch" />
        <ControlButtons @reset="handleReset" />
    </div>
</template>

<script>
import EpochIndicator from "@/components/EpochIndicator.vue";
import ControlButtons from "@/components/ControlButtons.vue";

export default {
    components: {
        EpochIndicator,
        ControlButtons
    },
    data() {
        return {
            epoch: 0
        }
    },
    mounted() {
        this.$options.sockets.onmessage = (msg) => {
            const data = JSON.parse(msg.data);
            if (data.epoch !== undefined) {
                this.epoch = data.epoch;
            }
        }
    },
    methods: {
        handleReset() {
            this.epoch = 0;
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
