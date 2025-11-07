import * as vscode from 'vscode';
import axios from 'axios';

interface Suggestion {
    id: string;
    content: string;
    type: string;
    confidence: number;
    priority: string;
}

export class MemoryAIProvider {
    private apiUrl: string;
    private userId: string;
    private isHaloMode: boolean = true;
    private suggestionTimeout: number;

    constructor() {
        const config = vscode.workspace.getConfiguration('memoryai');
        this.apiUrl = config.get('apiUrl', 'http://localhost:8082');
        this.userId = config.get('userId', 'vscode_user');
        this.suggestionTimeout = config.get('suggestionTimeout', 1200);
    }

    async getSmartSuggestion(): Promise<void> {
        try {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showInformationMessage('No active editor found');
                return;
            }

            const document = editor.document;
            const selection = editor.selection;
            const context = document.getText(selection);

            // Show loading
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'MemoryAI: Getting smart suggestion...',
                cancellable: false
            }, async (progress) => {
                try {
                    const response = await axios.post(`${this.apiUrl}/suggest`, {
                        userId: this.userId,
                        context: context,
                        document: document.getText(),
                        language: document.languageId,
                        line: editor.selection.active.line,
                        column: editor.selection.active.character,
                    }, {
                        timeout: this.suggestionTimeout
                    });

                    const suggestions: Suggestion[] = response.data.suggestions;
                    
                    if (suggestions.length > 0) {
                        this.showSuggestionQuickPick(suggestions, editor);
                    } else {
                        vscode.window.showInformationMessage('No suggestions available right now');
                    }
                } catch (error) {
                    console.error('Error getting suggestion:', error);
                    vscode.window.showErrorMessage('Failed to get suggestion. Check your connection.');
                }
            });

        } catch (error) {
            console.error('Error in getSmartSuggestion:', error);
            vscode.window.showErrorMessage('Failed to get smart suggestion');
        }
    }

    private showSuggestionQuickPick(suggestions: Suggestion[], editor: vscode.TextEditor): void {
        const items = suggestions.map(suggestion => ({
            label: suggestion.content,
            description: `${suggestion.type} (${(suggestion.confidence * 100).toFixed(0)}%)`,
            detail: `Priority: ${suggestion.priority}`,
            suggestion: suggestion
        }));

        vscode.window.showQuickPick(items, {
            placeHolder: 'Choose a suggestion',
            canPickMany: false
        }).then(async (selected) => {
            if (selected && selected.suggestion) {
                await this.applySuggestion(selected.suggestion, editor);
            }
        });
    }

    private async applySuggestion(suggestion: Suggestion, editor: vscode.TextEditor): Promise<void> {
        const document = editor.document;
        const selection = editor.selection;
        
        // Show diff preview if enabled
        const config = vscode.workspace.getConfiguration('memoryai');
        if (config.get('diffPreview', true)) {
            this.showDiffPreview(suggestion.content, editor);
        } else {
            // Apply directly
            await editor.edit(editBuilder => {
                editBuilder.replace(selection, suggestion.content);
            });
        }

        // Submit feedback
        try {
            await axios.post(`${this.apiUrl}/feedback`, {
                suggestionId: suggestion.id,
                accepted: true,
                userId: this.userId
            });
        } catch (error) {
            console.error('Error submitting feedback:', error);
        }
    }

    private showDiffPreview(newContent: string, editor: vscode.TextEditor): void {
        const originalContent = editor.document.getText(editor.selection);
        
        // Create diff URI
        const diffUri = vscode.Uri.parse(`memoryai-diff:/diff-preview`);
        
        // Show diff in new editor
        vscode.workspace.openTextDocument(diffUri).then(doc => {
            vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside, false);
        });
    }

    async searchMemoryGraph(): Promise<void> {
        const query = await vscode.window.showInputBox({
            prompt: 'Search your memory graph',
            placeHolder: 'What would you like to find?'
        });

        if (!query) return;

        try {
            const response = await axios.post(`${this.apiUrl}/search-memory`, {
                userId: this.userId,
                query: query
            });

            const results = response.data.results;
            
            if (results.length > 0) {
                const items = results.map((result: any) => ({
                    label: result.content,
                    description: result.type,
                    detail: `Relevance: ${(result.score * 100).toFixed(0)}%`,
                    result: result
                }));

                vscode.window.showQuickPick(items, {
                    placeHolder: 'Memory search results'
                }).then(selected => {
                    if (selected && selected.result) {
                        this.showMemoryDetails(selected.result);
                    }
                });
            } else {
                vscode.window.showInformationMessage('No memories found matching your query');
            }
        } catch (error) {
            console.error('Error searching memory:', error);
            vscode.window.showErrorMessage('Failed to search memory graph');
        }
    }

    private showMemoryDetails(memory: any): void {
        const panel = vscode.window.createWebviewPanel(
            'memoryDetails',
            `Memory: ${memory.content.substring(0, 20)}...`,
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        panel.webview.html = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {
                        font-family: var(--vscode-font-family);
                        padding: 20px;
                        background-color: var(--vscode-editor-background);
                        color: var(--vscode-editor-foreground);
                    }
                    .memory-content {
                        background-color: var(--vscode-editor-selectionBackground);
                        padding: 16px;
                        border-radius: 8px;
                        margin: 12px 0;
                    }
                    .memory-meta {
                        font-size: 0.9em;
                        color: var(--vscode-descriptionForeground);
                        margin-bottom: 8px;
                    }
                    .connections {
                        margin-top: 16px;
                    }
                    .connection-item {
                        background-color: var(--vscode-button-background);
                        color: var(--vscode-button-foreground);
                        padding: 4px 8px;
                        border-radius: 4px;
                        margin: 4px;
                        display: inline-block;
                    }
                </style>
            </head>
            <body>
                <h2>Memory Details</h2>
                <div class="memory-meta">
                    Type: ${memory.type} | Score: ${(memory.score * 100).toFixed(0)}%
                </div>
                <div class="memory-content">
                    ${memory.content}
                </div>
                ${memory.connections && memory.connections.length > 0 ? `
                    <div class="connections">
                        <h3>Connected Memories:</h3>
                        ${memory.connections.map((conn: string) => 
                            `<span class="connection-item">${conn}</span>`
                        ).join('')}
                    </div>
                ` : ''}
            </body>
            </html>
        `;
    }

    toggleMode(): void {
        this.isHaloMode = !this.isHaloMode;
        const mode = this.isHaloMode ? 'Halo' : 'Focus';
        vscode.window.showInformationMessage(`Switched to ${mode} mode`);
    }

    async showMemoryGraph(): Promise<void> {
        try {
            const panel = vscode.window.createWebviewPanel(
                'memoryGraph',
                'Memory Graph',
                vscode.ViewColumn.One,
                {
                    enableScripts: true,
                    retainContextWhenHidden: true
                }
            );

            // Get memory graph data
            const response = await axios.post(`${this.apiUrl}/memory-graph`, {
                userId: this.userId
            });

            const graphData = response.data;

            panel.webview.html = this.getMemoryGraphHTML(graphData);

        } catch (error) {
            console.error('Error showing memory graph:', error);
            vscode.window.showErrorMessage('Failed to load memory graph');
        }
    }

    private getMemoryGraphHTML(graphData: any): string {
        return `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {
                        margin: 0;
                        padding: 20px;
                        font-family: var(--vscode-font-family);
                        background-color: var(--vscode-editor-background);
                        color: var(--vscode-editor-foreground);
                    }
                    #graph-container {
                        width: 100%;
                        height: 600px;
                        border: 1px solid var(--vscode-panel-border);
                        border-radius: 8px;
                    }
                    .controls {
                        margin-bottom: 16px;
                    }
                    button {
                        background-color: var(--vscode-button-background);
                        color: var(--vscode-button-foreground);
                        border: none;
                        padding: 8px 16px;
                        margin: 4px;
                        border-radius: 4px;
                        cursor: pointer;
                    }
                    button:hover {
                        background-color: var(--vscode-button-hoverBackground);
                    }
                </style>
            </head>
            <body>
                <h1>Memory Graph</h1>
                <div class="controls">
                    <button onclick="resetView()">Reset View</button>
                    <button onclick="togglePhysics()">Toggle Physics</button>
                    <button onclick="exportGraph()">Export</button>
                </div>
                <div id="graph-container"></div>
                
                <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <script>
                    const nodes = new vis.DataSet(${JSON.stringify(graphData.nodes)});
                    const edges = new vis.DataSet(${JSON.stringify(graphData.connections)});
                    
                    const container = document.getElementById('graph-container');
                    const data = { nodes: nodes, edges: edges };
                    
                    const options = {
                        nodes: {
                            shape: 'dot',
                            size: 16,
                            font: {
                                size: 12,
                                color: '#ffffff'
                            },
                            borderWidth: 2,
                            color: {
                                background: '#6366F1',
                                border: '#8B5CF6'
                            }
                        },
                        edges: {
                            width: 2,
                            color: { color: '#14B8A6' },
                            smooth: {
                                type: 'continuous'
                            }
                        },
                        physics: {
                            enabled: true,
                            stabilization: {
                                enabled: true,
                                iterations: 1000
                            }
                        },
                        interaction: {
                            hover: true,
                            tooltipDelay: 200
                        }
                    };
                    
                    const network = new vis.Network(container, data, options);
                    
                    function resetView() {
                        network.fit();
                    }
                    
                    function togglePhysics() {
                        const isEnabled = network.physics.options.enabled;
                        network.setOptions({ physics: { enabled: !isEnabled } });
                    }
                    
                    function exportGraph() {
                        const canvas = container.querySelector('canvas');
                        const link = document.createElement('a');
                        link.download = 'memory-graph.png';
                        link.href = canvas.toDataURL();
                        link.click();
                    }
                    
                    // Handle node click
                    network.on('click', function(params) {
                        if (params.nodes.length > 0) {
                            const nodeId = params.nodes[0];
                            const node = nodes.get(nodeId);
                            vscode.postMessage({
                                command: 'showNodeDetails',
                                nodeId: nodeId,
                                node: node
                            });
                        }
                    });
                </script>
            </body>
            </html>
        `;
    }

    async generateCode(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showInformationMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const prompt = await vscode.window.showInputBox({
            prompt: 'Describe the code you want to generate',
            placeHolder: 'e.g., Create a function to sort an array'
        });

        if (!prompt) return;

        try {
            const response = await axios.post(`${this.apiUrl}/generate`, {
                userId: this.userId,
                prompt: prompt,
                language: editor.document.languageId,
                context: editor.document.getText(selection)
            });

            const generatedCode = response.data.code;
            
            if (vscode.workspace.getConfiguration('memoryai').get('diffPreview', true)) {
                this.showCodePreview(generatedCode, editor, 'Generated Code');
            } else {
                await editor.edit(editBuilder => {
                    editBuilder.replace(selection, generatedCode);
                });
            }

        } catch (error) {
            console.error('Error generating code:', error);
            vscode.window.showErrorMessage('Failed to generate code');
        }
    }

    async explainCode(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showInformationMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const code = editor.document.getText(selection);

        if (!code) {
            vscode.window.showInformationMessage('Please select code to explain');
            return;
        }

        try {
            const response = await axios.post(`${this.apiUrl}/explain`, {
                userId: this.userId,
                code: code,
                language: editor.document.languageId
            });

            const explanation = response.data.explanation;
            
            const panel = vscode.window.createWebviewPanel(
                'codeExplanation',
                'Code Explanation',
                vscode.ViewColumn.Beside,
                {}
            );

            panel.webview.html = `
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body {
                            font-family: var(--vscode-font-family);
                            padding: 20px;
                            background-color: var(--vscode-editor-background);
                            color: var(--vscode-editor-foreground);
                        }
                        .code-block {
                            background-color: var(--vscode-textCodeBlock-background);
                            padding: 12px;
                            border-radius: 6px;
                            margin: 12px 0;
                            font-family: var(--vscode-editor-font-family);
                        }
                        .explanation {
                            line-height: 1.6;
                        }
                    </style>
                </head>
                <body>
                    <h2>Code Explanation</h2>
                    <div class="code-block">${code}</div>
                    <div class="explanation">${explanation}</div>
                </body>
                </html>
            `;

        } catch (error) {
            console.error('Error explaining code:', error);
            vscode.window.showErrorMessage('Failed to explain code');
        }
    }

    async optimizeCode(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showInformationMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const code = editor.document.getText(selection);

        if (!code) {
            vscode.window.showInformationMessage('Please select code to optimize');
            return;
        }

        try {
            const response = await axios.post(`${this.apiUrl}/optimize`, {
                userId: this.userId,
                code: code,
                language: editor.document.languageId
            });

            const optimizedCode = response.data.optimizedCode;
            const suggestions = response.data.suggestions;

            this.showCodePreview(optimizedCode, editor, 'Optimized Code', suggestions);

        } catch (error) {
            console.error('Error optimizing code:', error);
            vscode.window.showErrorMessage('Failed to optimize code');
        }
    }

    private showCodePreview(
        newCode: string, 
        editor: vscode.TextEditor, 
        title: string, 
        suggestions?: string[]
    ): void {
        const originalCode = editor.document.getText(editor.selection);
        
        const panel = vscode.window.createWebviewPanel(
            'codePreview',
            title,
            vscode.ViewColumn.Beside,
            {
                enableScripts: true
            }
        );

        panel.webview.html = `
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {
                        font-family: var(--vscode-font-family);
                        padding: 20px;
                        background-color: var(--vscode-editor-background);
                        color: var(--vscode-editor-foreground);
                    }
                    .code-container {
                        margin: 12px 0;
                    }
                    .code-label {
                        font-weight: 600;
                        margin-bottom: 8px;
                        color: var(--vscode-editor-foreground);
                    }
                    .code-block {
                        background-color: var(--vscode-textCodeBlock-background);
                        padding: 12px;
                        border-radius: 6px;
                        font-family: var(--vscode-editor-font-family);
                        white-space: pre-wrap;
                        overflow-x: auto;
                    }
                    .suggestions {
                        margin-top: 16px;
                        padding: 12px;
                        background-color: var(--vscode-editor-selectionBackground);
                        border-radius: 6px;
                    }
                    .suggestion-item {
                        margin: 4px 0;
                        padding: 4px 8px;
                        background-color: var(--vscode-button-background);
                        color: var(--vscode-button-foreground);
                        border-radius: 4px;
                    }
                    .actions {
                        margin-top: 20px;
                        display: flex;
                        gap: 8px;
                    }
                    button {
                        background-color: var(--vscode-button-background);
                        color: var(--vscode-button-foreground);
                        border: none;
                        padding: 8px 16px;
                        border-radius: 4px;
                        cursor: pointer;
                    }
                    button:hover {
                        background-color: var(--vscode-button-hoverBackground);
                    }
                    .accept {
                        background-color: #10B981;
                    }
                    .reject {
                        background-color: #EF4444;
                    }
                </style>
            </head>
            <body>
                <h2>${title}</h2>
                
                <div class="code-container">
                    <div class="code-label">Original:</div>
                    <div class="code-block">${originalCode}</div>
                </div>
                
                <div class="code-container">
                    <div class="code-label">${title}:</div>
                    <div class="code-block">${newCode}</div>
                </div>
                
                ${suggestions && suggestions.length > 0 ? `
                    <div class="suggestions">
                        <h3>Optimization Suggestions:</h3>
                        ${suggestions.map(s => `<div class="suggestion-item">${s}</div>`).join('')}
                    </div>
                ` : ''}
                
                <div class="actions">
                    <button class="accept" onclick="acceptCode()">Accept</button>
                    <button class="reject" onclick="rejectCode()">Reject</button>
                    <button onclick="copyCode()">Copy</button>
                </div>
                
                <script>
                    const vscode = acquireVsCodeApi();
                    const newCode = ${JSON.stringify(newCode)};
                    
                    function acceptCode() {
                        vscode.postMessage({ command: 'accept', code: newCode });
                    }
                    
                    function rejectCode() {
                        vscode.postMessage({ command: 'reject' });
                    }
                    
                    function copyCode() {
                        navigator.clipboard.writeText(newCode);
                        vscode.postMessage({ command: 'copied' });
                    }
                </script>
            </body>
            </html>
        `;

        // Handle messages from webview
        panel.webview.onDidReceiveMessage(async (message) => {
            switch (message.command) {
                case 'accept':
                    await editor.edit(editBuilder => {
                        editBuilder.replace(editor.selection, message.code);
                    });
                    panel.dispose();
                    break;
                case 'reject':
                    panel.dispose();
                    break;
                case 'copied':
                    vscode.window.showInformationMessage('Code copied to clipboard');
                    break;
            }
        });
    }

    onDocumentChange(event: vscode.TextDocumentChangeEvent): void {
        // Debounce document changes
        // Implementation would include debouncing logic
        console.log('Document changed:', event.document.fileName);
    }

    onEditorChange(editor: vscode.TextEditor): void {
        console.log('Editor changed:', editor.document.fileName);
    }

    onConfigurationChange(): void {
        const config = vscode.workspace.getConfiguration('memoryai');
        this.apiUrl = config.get('apiUrl', 'http://localhost:8082');
        this.userId = config.get('userId', 'vscode_user');
        this.suggestionTimeout = config.get('suggestionTimeout', 1200);
        
        console.log('Configuration updated');
    }
}