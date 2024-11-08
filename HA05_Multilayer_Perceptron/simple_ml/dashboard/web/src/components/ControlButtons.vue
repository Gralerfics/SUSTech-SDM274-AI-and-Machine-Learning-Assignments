<template>
    <div class="control-buttons">
        <button class="control-button" @click="handlePlayPause">
            <i class="material-icons">
                {{ state === 'running' ? 'pause' : 'play_arrow' }}
            </i>
        </button>
        <button class="control-button reset" @click="handleReset">
            <i class="material-icons">replay</i>
        </button>
    </div>
</template>

<script>
import axios from 'axios';

import global_config from '@/config.js'

export default {
    data() {
        return {
            state: 'stopped',
            task_id: null
        };
    },
    mounted() {
        this.fetchState(); // fetch the core state when mounted
    },
    methods: {
        async fetchState() {
            try {
                const response = await axios.get(new URL("/api/get_state", global_config.http_address).href);
                if (response.data.status === 'ok') {
                    this.state = response.data.state.state;
                    }
            } catch (error) {
                console.error("Failed to fetch state:", error);
            }
        },
        async handlePlayPause() {
            try {
                let response = null;
                if (this.state === 'stopped') {
                    response = await axios.post(
                        new URL("/api/launch", global_config.http_address).href,
                        {
                            dataset: {

                            },
                            model: {

                            },
                            loss: {

                            },
                            optimizer: {

                            }
                        } // TODO
                    );
                } else if (this.state === 'paused') {
                    response = await axios.get(new URL("/api/resume", global_config.http_address).href);
                } else if (this.state === 'running') {
                    response = await axios.get(new URL("/api/pause", global_config.http_address).href);
                }
                if (response !== null && response.data.status === 'ok') {
                    this.state = response.data.state.state;
                    this.task_id = (response.data.state.task_id !== undefined) ? response.data.state.task_id : null;
                }
            } catch (error) {
                console.error("Play/Pause action failed:", error);
            }
        },
        async handleReset() {
            try {
                const response = await axios.get(new URL("/api/stop", global_config.http_address).href);
                this.state = 'stopped';
                this.$emit('reset');
            } catch (error) {
                console.error("Reset action failed:", error);
            }
        }
    }
};
</script>

<style scoped>
.control-buttons {
    display: flex;
    gap: 10px;
}

.control-button {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background-color: #183D4E;
    color: white;
    font-size: 24px;
    display: flex;
    justify-content: center;
    align-items: center;
    border: none;
    cursor: pointer;
    outline: none;
}

.control-button.play::before {
    content: '▶';
}

.control-button.pause::before {
    content: '||';
}

.reset {
    background-color: #183D4E;
    font-size: 20px;
}
</style>