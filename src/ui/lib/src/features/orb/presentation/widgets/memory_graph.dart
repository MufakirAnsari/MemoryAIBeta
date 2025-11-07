import 'package:flutter/material.dart';
import 'package:glassmorphism/glassmorphism.dart';
import 'package:memoryai_ui/src/core/theme/app_theme.dart';

class MemoryGraph extends StatefulWidget {
  final VoidCallback onClose;
  
  const MemoryGraph({
    super.key,
    required this.onClose,
  });

  @override
  State<MemoryGraph> createState() => _MemoryGraphState();
}

class _MemoryGraphState extends State<MemoryGraph>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  
  final List<MemoryNode> _nodes = [
    MemoryNode(id: '1', content: 'Python Programming', x: 0.2, y: 0.3, connections: ['2', '3']),
    MemoryNode(id: '2', content: 'Machine Learning', x: 0.6, y: 0.2, connections: ['1', '4']),
    MemoryNode(id: '3', content: 'Web Development', x: 0.3, y: 0.7, connections: ['1', '5']),
    MemoryNode(id: '4', content: 'Data Science', x: 0.8, y: 0.5, connections: ['2', '6']),
    MemoryNode(id: '5', content: 'JavaScript', x: 0.1, y: 0.8, connections: ['3']),
    MemoryNode(id: '6', content: 'Statistics', x: 0.7, y: 0.8, connections: ['4']),
  ];
  
  MemoryNode? _selectedNode;
  
  @override
  void initState() {
    super.initState();
    
    _controller = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    );
    
    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOut,
    ));
    
    _controller.forward();
  }
  
  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _fadeAnimation,
      builder: (context, child) {
        return Opacity(
          opacity: _fadeAnimation.value,
          child: child,
        );
      },
      child: GestureDetector(
        onTap: widget.onClose,
        child: Container(
          color: Colors.black.withOpacity(0.5),
          child: Center(
            child: GestureDetector(
              onTap: () {}, // Prevent closing when tapping on the graph
              child: Container(
                width: MediaQuery.of(context).size.width * 0.9,
                height: MediaQuery.of(context).size.height * 0.8,
                padding: const EdgeInsets.all(20),
                child: Glassmorphism(
                  blur: 25,
                  opacity: 0.15,
                  radius: 20,
                  child: Container(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      children: [
                        _buildHeader(),
                        const SizedBox(height: 20),
                        Expanded(
                          child: _buildGraphView(),
                        ),
                        if (_selectedNode != null)
                          _buildNodeDetails(),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
  
  Widget _buildHeader() {
    return Row(
      children: [
        const Icon(
          Icons.bubble_chart,
          color: AppTheme.teenPurple,
          size: 24,
        ),
        const SizedBox(width: 8),
        const Text(
          'Memory Graph',
          style: TextStyle(
            color: Colors.white,
            fontSize: 20,
            fontWeight: FontWeight.w600,
          ),
        ),
        const Spacer(),
        IconButton(
          icon: const Icon(Icons.close, color: Colors.white),
          onPressed: widget.onClose,
          tooltip: 'Close',
        ),
      ],
    );
  }
  
  Widget _buildGraphView() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Stack(
        children: [
          // Connections
          ..._buildConnections(),
          // Nodes
          ..._buildNodes(),
        ],
      ),
    );
  }
  
  List<Widget> _buildConnections() {
    List<Widget> connections = [];
    
    for (final node in _nodes) {
      for (final connectionId in node.connections) {
        final connectedNode = _nodes.firstWhere((n) => n.id == connectionId);
        
        connections.add(
          CustomPaint(
            painter: ConnectionPainter(
              start: Offset(
                node.x * MediaQuery.of(context).size.width * 0.7,
                node.y * MediaQuery.of(context).size.height * 0.5,
              ),
              end: Offset(
                connectedNode.x * MediaQuery.of(context).size.width * 0.7,
                connectedNode.y * MediaQuery.of(context).size.height * 0.5,
              ),
              color: AppTheme.teenTeal.withOpacity(0.3),
              strokeWidth: 2,
            ),
          ),
        );
      }
    }
    
    return connections;
  }
  
  List<Widget> _buildNodes() {
    return _nodes.map((node) {
      return Positioned(
        left: node.x * MediaQuery.of(context).size.width * 0.7 - 40,
        top: node.y * MediaQuery.of(context).size.height * 0.5 - 20,
        child: GestureDetector(
          onTap: () => _selectNode(node),
          child: _buildNodeWidget(node),
        ),
      );
    }).toList();
  }
  
  Widget _buildNodeWidget(MemoryNode node) {
    final isSelected = _selectedNode?.id == node.id;
    
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      width: isSelected ? 100 : 80,
      height: isSelected ? 60 : 40,
      child: Glassmorphism(
        blur: 15,
        opacity: isSelected ? 0.3 : 0.2,
        radius: 12,
        child: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            border: Border.all(
              color: isSelected 
                  ? AppTheme.teenOrange 
                  : AppTheme.teenPurple.withOpacity(0.5),
              width: isSelected ? 2 : 1,
            ),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Center(
            child: Text(
              node.content,
              style: TextStyle(
                color: Colors.white,
                fontSize: isSelected ? 11 : 10,
                fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
              ),
              textAlign: TextAlign.center,
              overflow: TextOverflow.ellipsis,
              maxLines: 2,
            ),
          ),
        ),
      ),
    );
  }
  
  Widget _buildNodeDetails() {
    if (_selectedNode == null) return const SizedBox.shrink();
    
    return Container(
      margin: const EdgeInsets.only(top: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            _selectedNode!.content,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Connections: ${_selectedNode!.connections.length}',
            style: TextStyle(
              color: Colors.white.withOpacity(0.7),
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            children: _selectedNode!.connections.map((id) {
              final connectedNode = _nodes.firstWhere((n) => n.id == id);
              return Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: AppTheme.teenPurple.withOpacity(0.3),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  connectedNode.content,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                  ),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }
  
  void _selectNode(MemoryNode node) {
    setState(() {
      if (_selectedNode?.id == node.id) {
        _selectedNode = null;
      } else {
        _selectedNode = node;
      }
    });
  }
}

class ConnectionPainter extends CustomPainter {
  final Offset start;
  final Offset end;
  final Color color;
  final double strokeWidth;
  
  ConnectionPainter({
    required this.start,
    required this.end,
    required this.color,
    required this.strokeWidth,
  });
  
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke;
    
    canvas.drawLine(start, end, paint);
  }
  
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

@immutable
class MemoryNode {
  final String id;
  final String content;
  final double x;
  final double y;
  final List<String> connections;
  
  const MemoryNode({
    required this.id,
    required this.content,
    required this.x,
    required this.y,
    required this.connections,
  });
}