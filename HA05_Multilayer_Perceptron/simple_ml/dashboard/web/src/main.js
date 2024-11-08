import { createApp } from 'vue';

import App from './App.vue';
import router from './router';
import VueNativeSock from 'vue-native-websocket-vue3';

import global_config from "@/config.js";

const app = createApp(App);

app.use(router);
app.use(VueNativeSock, (new URL("/notify", global_config.ws_address)).href, {
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 3000
});

app.mount('#app');
