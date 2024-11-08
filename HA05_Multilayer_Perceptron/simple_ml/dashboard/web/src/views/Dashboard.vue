<template>
    <div class="dashboard">
        <TwoFeaturesClassificationModelOutputChart />
    </div>
</template>

<script>
import TwoFeaturesClassificationModelOutputChart from '../components/TwoFeaturesClassificationModelOutputChart.vue';

// import axios from 'axios';
// import global_config from "@/config.js";

export default {
    components: {
        TwoFeaturesClassificationModelOutputChart
    },
    mounted() {

    },
    methods: {

    }
};
</script>

<style>
.dashboard {
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 20px;
}
</style>
