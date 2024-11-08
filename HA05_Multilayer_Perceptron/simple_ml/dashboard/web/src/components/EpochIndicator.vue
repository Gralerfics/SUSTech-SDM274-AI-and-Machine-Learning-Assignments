<template>
    <div class="epoch-indicator">
        {{ formattedEpoch }}
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
            return this.epoch.toString().padStart(6, '0').replace(/\B(?=(\d{3})+(?!\d))/g, ',');
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