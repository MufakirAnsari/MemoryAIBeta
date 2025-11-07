import * as vscode from 'vscode';
import { MemoryAIProvider } from './memoryaiProvider';
import { SuggestionProvider } from './suggestionProvider';
import { DiffPreviewProvider } from './diffPreviewProvider';
import { StatusBarManager } from './statusBarManager';

export function activate(context: vscode.ExtensionContext) {
    const provider = new MemoryAIProvider();
    const suggestionProvider = new SuggestionProvider();
    const diffPreviewProvider = new DiffPreviewProvider();
    const statusBarManager = new StatusBarManager();

    // Register commands
    const commands = [
        vscode.commands.registerCommand('memoryai.suggest', () => {
            provider.getSmartSuggestion();
        }),

        vscode.commands.registerCommand('memoryai.searchMemory', () => {
            provider.searchMemoryGraph();
        }),

        vscode.commands.registerCommand('memoryai.toggleMode', () => {
            provider.toggleMode();
            statusBarManager.updateMode();
        }),

        vscode.commands.registerCommand('memoryai.showMemoryGraph', () => {
            provider.showMemoryGraph();
        }),

        vscode.commands.registerCommand('memoryai.generateCode', () => {
            provider.generateCode();
        }),

        vscode.commands.registerCommand('memoryai.explainCode', () => {
            provider.explainCode();
        }),

        vscode.commands.registerCommand('memoryai.optimizeCode', () => {
            provider.optimizeCode();
        }),
    ];

    // Register providers
    const providers = [
        vscode.languages.registerCompletionItemProvider(
            ['python', 'javascript', 'typescript'],
            suggestionProvider,
            '.'
        ),

        vscode.workspace.registerTextDocumentContentProvider(
            'memoryai-diff',
            diffPreviewProvider
        ),
    ];

    // Register event listeners
    const listeners = [
        vscode.workspace.onDidChangeTextDocument((event) => {
            if (vscode.workspace.getConfiguration('memoryai').get('autoSuggest')) {
                provider.onDocumentChange(event);
            }
        }),

        vscode.window.onDidChangeActiveTextEditor((editor) => {
            if (editor) {
                provider.onEditorChange(editor);
            }
        }),

        vscode.workspace.onDidChangeConfiguration((event) => {
            if (event.affectsConfiguration('memoryai')) {
                provider.onConfigurationChange();
            }
        }),
    ];

    // Initialize status bar
    statusBarManager.initialize();

    // Add all disposables to context
    context.subscriptions.push(...commands, ...providers, ...listeners);
    context.subscriptions.push(statusBarManager);

    // Show welcome message
    vscode.window.showInformationMessage(
        '🧠 MemoryAI Enterprise activated! Your AI companion is ready.',
        'Get Started',
        'Settings'
    ).then((selection) => {
        if (selection === 'Get Started') {
            vscode.commands.executeCommand('memoryai.suggest');
        } else if (selection === 'Settings') {
            vscode.commands.executeCommand('workbench.action.openSettings', 'memoryai');
        }
    });

    console.log('MemoryAI Enterprise extension activated');
}

export function deactivate() {
    console.log('MemoryAI Enterprise extension deactivated');
}