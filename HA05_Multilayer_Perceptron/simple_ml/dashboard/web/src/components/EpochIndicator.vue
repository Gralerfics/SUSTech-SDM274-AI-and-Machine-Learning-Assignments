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
            const epoch_num = (this.isValidData(this.epoch)) ? this.epoch : 0;
            return epoch_num.toString().padStart(6, '0').replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        }
    },
    methods: {
        isValidData(data) {
            return (data !== null && data !== undefined);
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