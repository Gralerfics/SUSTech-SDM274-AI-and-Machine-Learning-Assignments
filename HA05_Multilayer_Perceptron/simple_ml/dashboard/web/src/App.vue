<script setup>
import { RouterView, useRoute } from 'vue-router';

import Navbar from '@/components/NavigationBar.vue';

const route = useRoute();
</script>

<template>
    <div id="body_container">
        <Navbar v-if="!route.meta?.noNavbar" />
        <RouterView />
    </div>

    <div v-if="!isWebsocketConnected" class="overlay">
        <div class="overlay-panel">
            Connecting ...
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            isWebsocketConnected: false
        }
    },
    mounted() {
        this.$options.sockets.onopen = () => {
            this.isWebsocketConnected = true;
        }

        this.$options.sockets.onclose = () => {
            this.isWebsocketConnected = false;
        }
    }
}
</script>

<style>
@import "@/assets/main.css";

.overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}

.overlay-panel {
    padding: 20px;
    background-color: white;
    border-radius: 8px;
    font-size: 18px;
    color: #333;
    text-align: center;
}
</style>