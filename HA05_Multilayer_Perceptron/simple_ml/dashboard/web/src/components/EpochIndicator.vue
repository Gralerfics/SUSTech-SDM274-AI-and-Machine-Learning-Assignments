<template>
    <div>
        <div style="color: #969696">
            Epoch
        </div>
        <div class="epoch-indicator">
            {{ formattedEpoch }}
        </div>
    </div>
</template>

<script>
export default {
    props: {
        epoch: {
            type: Number,
            required: true
        }
    },
    computed: {
        formattedEpoch() {
            const epoch_num = (this.epoch !== null && this.epoch !== undefined) ? this.epoch : 0;
            return epoch_num.toString().padStart(6, '0').replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        }
    }
};
</script>

<style scoped>
.epoch-indicator {
    font-size: 2em;
    font-weight: bold;
    color: #333;
}
</style>