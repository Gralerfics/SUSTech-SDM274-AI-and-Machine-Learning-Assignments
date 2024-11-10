<template>
    <div class="model-builder">
        <div class="building-zone">
            <button @click="addBlock('Linear')" class="add-block-btn">Add Linear Layer</button>
            <button @click="addBlock('Sigmoid')" class="add-block-btn">Add Sigmoid Layer</button>
            <button @click="addBlock('ReLU')" class="add-block-btn">Add ReLU Layer</button>

            <VueDraggable
                v-model="localModelBlocks"
                animation="150"
                ghostClass="ghost"
                group="blocks"
                @update="onUpdate"
                handle=".block-header"
            >
                <div
                    v-for="(block, index) in localModelBlocks"
                    :key="index"
                    class="building-block"
                >
                    <div class="block-header">
                        <b>{{ block.type }}</b>
                        <button @click.stop="removeBlock(index)" class="remove-btn">Remove</button>
                    </div>
                    <div v-if="block.params !== undefined && Object.keys(block.params).length > 0">
                        <hr />
                        <div v-for="(value, key) in block.params" class="params" :key="key">
                            <div class="param-field">
                                <label>{{ key }}</label>
                                <input type="number" v-model="block.params[key]" min="1" @click.stop @focus.stop />
                            </div>
                        </div>
                    </div>
                </div>
            </VueDraggable>
        </div>
    </div>
</template>

<script>
import { VueDraggable } from 'vue-draggable-plus'

export default {
    components: {
        VueDraggable
    },
    props: {
        modelBlocks: {
            type: Array,
            required: true
        }
    },
    data() {
        return {
            localModelBlocks: JSON.parse(JSON.stringify(this.modelBlocks)) // deep copy
        }
    },
    watch: {
        localModelBlocks: {
            handler(newVal) {
                this.$emit('update:modelBlocks', newVal)
            },
            deep: true
        }
    },
    methods: {
        addBlock(type) {
            let newBlock = {}
            switch (type) {
                case 'Linear':
                    newBlock = { type: 'Linear', params: { in_dim: 1, out_dim: 1 } }
                    break
                case 'Sigmoid':
                    newBlock = { type: 'Sigmoid', params: {} }
                    break
                case 'ReLU':
                    newBlock = { type: 'ReLU', params: {} }
                    break
            }
            this.localModelBlocks.push(newBlock)
        },
        removeBlock(index) {
            this.localModelBlocks.splice(index, 1)
        },
        onUpdate() {
            // TODO
        }
    }
};
</script>

<style scoped>
.model-builder {
    display: flex;
    flex-direction: column;
    width: 600px;
    border: 1px solid #ccc;
}

.building-zone {
    width: 100%;
    margin: 10px;
}

.add-block-btn {
    padding: 10px;
    margin: 5px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}

.add-block-btn:hover {
    background-color: #45a049;
}

.building-block {
    background-color: #f0f0f0;
    padding: 10px;
    width: calc(100% - 50px);
    margin: 5px;
    border-radius: 5px;
}

.block-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: move;
}

.remove-btn {
    background-color: red;
    color: white;
    padding: 5px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}

.params {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 10px;
}

.param-field {
    display: flex;
    align-items: center;
}

.param-field label {
    flex: 1;
    font-style: italic;
    color: #555;
}

.param-field input {
    flex: 3;
    padding: 5px;
    border: none;
    border-radius: 8px;
    background-color: #fff;
    box-shadow: 0px 1px 3px rgba(0, 0, 0, 0.1);
    width: 60%;
    text-align: right;
}
</style>