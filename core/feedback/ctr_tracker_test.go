package feedback

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMemoryCTRTracker(t *testing.T) {
	ctx := context.Background()
	tracker := NewMemoryCTRTracker()

	t.Run("RecordImpression", func(t *testing.T) {
		impression := Impression{
			QueryID:      "query1",
			Query:        "test query",
			DocumentID:   "doc1",
			Position:     1,
			UserID:       "user1",
			SessionID:    "session1",
			Timestamp:    time.Now(),
			ResultCount:  10,
			SearchMode:   "hybrid",
		}

		err := tracker.RecordImpression(ctx, impression)
		assert.NoError(t, err)
	})

	t.Run("RecordClick", func(t *testing.T) {
		// First record an impression
		impression := Impression{
			QueryID:    "query2",
			Query:      "test query 2",
			DocumentID: "doc2",
			Position:   2,
			UserID:     "user1",
			SessionID:  "session1",
			Timestamp:  time.Now(),
		}
		err := tracker.RecordImpression(ctx, impression)
		require.NoError(t, err)

		// Then record a click
		click := Click{
			QueryID:    "query2",
			DocumentID: "doc2",
			Position:   2,
			UserID:     "user1",
			SessionID:  "session1",
			Timestamp:  time.Now(),
			DwellTime:  45.5,
		}
		err = tracker.RecordClick(ctx, click)
		assert.NoError(t, err)
	})

	t.Run("GetDocumentCTR", func(t *testing.T) {
		// Record multiple impressions and clicks for a document
		for i := 0; i < 10; i++ {
			impression := Impression{
				QueryID:    "query" + string(rune('1'+i)),
				DocumentID: "popular_doc",
				Position:   1,
				UserID:     "user" + string(rune('1'+i)),
				SessionID:  "session" + string(rune('1'+i)),
				Timestamp:  time.Now(),
			}
			err := tracker.RecordImpression(ctx, impression)
			require.NoError(t, err)

			// Click on 3 out of 10 impressions
			if i < 3 {
				click := Click{
					QueryID:    impression.QueryID,
					DocumentID: impression.DocumentID,
					Position:   impression.Position,
					UserID:     impression.UserID,
					SessionID:  impression.SessionID,
					Timestamp:  time.Now(),
					DwellTime:  30.0 + float64(i*10),
				}
				err = tracker.RecordClick(ctx, click)
				require.NoError(t, err)
			}
		}

		// Get CTR metrics
		metrics, err := tracker.GetDocumentCTR(ctx, "popular_doc")
		assert.NoError(t, err)
		assert.Equal(t, int64(10), metrics.Impressions)
		assert.Equal(t, int64(3), metrics.Clicks)
		assert.Equal(t, 0.3, metrics.CTR)
		assert.Equal(t, 1.0, metrics.AvgPosition)
		assert.InDelta(t, 40.0, metrics.AvgDwellTime, 0.1) // (30+40+50)/3
	})

	t.Run("GetQueryCTR", func(t *testing.T) {
		// Record impressions and clicks for a specific query
		query := "machine learning"
		for i := 0; i < 5; i++ {
			impression := Impression{
				QueryID:    "ml_query",
				Query:      query,
				DocumentID: "doc" + string(rune('1'+i)),
				Position:   i + 1,
				UserID:     "user1",
				SessionID:  "session1",
				Timestamp:  time.Now(),
			}
			err := tracker.RecordImpression(ctx, impression)
			require.NoError(t, err)

			// Click on first 2 results
			if i < 2 {
				click := Click{
					QueryID:    impression.QueryID,
					DocumentID: impression.DocumentID,
					Position:   impression.Position,
					UserID:     impression.UserID,
					SessionID:  impression.SessionID,
					Timestamp:  time.Now(),
					DwellTime:  60.0,
				}
				err = tracker.RecordClick(ctx, click)
				require.NoError(t, err)
			}
		}

		// Get query CTR
		metrics, err := tracker.GetQueryCTR(ctx, query)
		assert.NoError(t, err)
		assert.Equal(t, int64(5), metrics.Impressions)
		assert.Equal(t, int64(2), metrics.Clicks)
		assert.Equal(t, 0.4, metrics.CTR)
		assert.Equal(t, 60.0, metrics.AvgDwellTime)
	})

	t.Run("GetPositionCTR", func(t *testing.T) {
		// Create a fresh tracker for this test to avoid data from previous tests
		freshTracker := NewMemoryCTRTracker()
		
		// Record clicks at different positions
		positions := []int{1, 1, 1, 2, 2, 3}
		clicks := []bool{true, true, false, true, false, false}

		for i, pos := range positions {
			impression := Impression{
				QueryID:    "pos_query" + string(rune('1'+i)),
				DocumentID: "pos_doc" + string(rune('1'+i)),
				Position:   pos,
				UserID:     "user1",
				SessionID:  "session1",
				Timestamp:  time.Now(),
			}
			err := freshTracker.RecordImpression(ctx, impression)
			require.NoError(t, err)

			if clicks[i] {
				click := Click{
					QueryID:    impression.QueryID,
					DocumentID: impression.DocumentID,
					Position:   impression.Position,
					UserID:     impression.UserID,
					SessionID:  impression.SessionID,
					Timestamp:  time.Now(),
				}
				err = freshTracker.RecordClick(ctx, click)
				require.NoError(t, err)
			}
		}

		// Check CTR by position
		pos1Metrics, err := freshTracker.GetPositionCTR(ctx, 1)
		assert.NoError(t, err)
		assert.Equal(t, int64(3), pos1Metrics.Impressions)
		assert.Equal(t, int64(2), pos1Metrics.Clicks)
		assert.InDelta(t, 0.667, pos1Metrics.CTR, 0.001)

		pos2Metrics, err := freshTracker.GetPositionCTR(ctx, 2)
		assert.NoError(t, err)
		assert.Equal(t, int64(2), pos2Metrics.Impressions)
		assert.Equal(t, int64(1), pos2Metrics.Clicks)
		assert.Equal(t, 0.5, pos2Metrics.CTR)

		pos3Metrics, err := freshTracker.GetPositionCTR(ctx, 3)
		assert.NoError(t, err)
		assert.Equal(t, int64(1), pos3Metrics.Impressions)
		assert.Equal(t, int64(0), pos3Metrics.Clicks)
		assert.Equal(t, 0.0, pos3Metrics.CTR)
	})

	t.Run("GetOptimalRanking", func(t *testing.T) {
		// Set up CTR data for different documents
		docs := []struct {
			id          string
			impressions int
			clicks      int
		}{
			{"doc_high_ctr", 100, 50},   // 50% CTR
			{"doc_med_ctr", 100, 20},    // 20% CTR
			{"doc_low_ctr", 100, 5},     // 5% CTR
			{"doc_no_data", 0, 0},       // No data
		}

		// Record impressions and clicks
		for _, doc := range docs {
			for i := 0; i < doc.impressions; i++ {
				impression := Impression{
					QueryID:    "ranking_query" + string(rune(i)),
					DocumentID: doc.id,
					Position:   1, // Same position for fair comparison
					UserID:     "user" + string(rune(i)),
					SessionID:  "session" + string(rune(i)),
					Timestamp:  time.Now(),
				}
				err := tracker.RecordImpression(ctx, impression)
				require.NoError(t, err)

				if i < doc.clicks {
					click := Click{
						QueryID:    impression.QueryID,
						DocumentID: impression.DocumentID,
						Position:   impression.Position,
						UserID:     impression.UserID,
						SessionID:  impression.SessionID,
						Timestamp:  time.Now(),
					}
					err = tracker.RecordClick(ctx, click)
					require.NoError(t, err)
				}
			}
		}

		// Get optimal ranking
		documentIDs := []string{"doc_low_ctr", "doc_no_data", "doc_high_ctr", "doc_med_ctr"}
		optimalRanking, err := tracker.GetOptimalRanking(ctx, documentIDs, "test query")
		assert.NoError(t, err)
		assert.Equal(t, []string{"doc_high_ctr", "doc_med_ctr", "doc_low_ctr", "doc_no_data"}, optimalRanking)
	})

	t.Run("ExportMetrics", func(t *testing.T) {
		// Add some test data
		for i := 0; i < 20; i++ {
			impression := Impression{
				QueryID:    "export_query" + string(rune(i%5)),
				Query:      "export test " + string(rune(i%3)),
				DocumentID: "export_doc" + string(rune(i%4)),
				Position:   (i % 5) + 1,
				UserID:     "export_user" + string(rune(i%2)),
				SessionID:  "export_session" + string(rune(i)),
				Timestamp:  time.Now(),
			}
			err := tracker.RecordImpression(ctx, impression)
			require.NoError(t, err)

			// Click on some impressions
			if i%3 == 0 {
				click := Click{
					QueryID:    impression.QueryID,
					DocumentID: impression.DocumentID,
					Position:   impression.Position,
					UserID:     impression.UserID,
					SessionID:  impression.SessionID,
					Timestamp:  time.Now(),
					DwellTime:  float64(10 + i*2),
				}
				err = tracker.RecordClick(ctx, click)
				require.NoError(t, err)
			}
		}

		// Export metrics
		report, err := tracker.ExportMetrics(ctx)
		assert.NoError(t, err)
		assert.Greater(t, report.TotalImpressions, int64(0))
		assert.Greater(t, report.TotalClicks, int64(0))
		assert.Greater(t, report.OverallCTR, 0.0)
		assert.NotEmpty(t, report.TopDocuments)
		assert.NotEmpty(t, report.PositionCTR)
		assert.NotNil(t, report.HourlyTrends)
	})

	t.Run("ConfidenceScore", func(t *testing.T) {
		// Record few impressions
		for i := 0; i < 3; i++ {
			impression := Impression{
				QueryID:    "conf_query" + string(rune(i)),
				DocumentID: "conf_doc_low",
				Position:   1,
				UserID:     "user" + string(rune(i)),
				SessionID:  "session" + string(rune(i)),
				Timestamp:  time.Now(),
			}
			err := tracker.RecordImpression(ctx, impression)
			require.NoError(t, err)
		}

		// Record many impressions
		for i := 0; i < 100; i++ {
			impression := Impression{
				QueryID:    "conf_query_high" + string(rune(i)),
				DocumentID: "conf_doc_high",
				Position:   1,
				UserID:     "user" + string(rune(i)),
				SessionID:  "session" + string(rune(i)),
				Timestamp:  time.Now(),
			}
			err := tracker.RecordImpression(ctx, impression)
			require.NoError(t, err)
		}

		// Compare confidence scores
		lowMetrics, err := tracker.GetDocumentCTR(ctx, "conf_doc_low")
		assert.NoError(t, err)
		assert.Less(t, lowMetrics.ConfidenceScore, 0.8)

		highMetrics, err := tracker.GetDocumentCTR(ctx, "conf_doc_high")
		assert.NoError(t, err)
		assert.Greater(t, highMetrics.ConfidenceScore, 0.95)
	})
}