-- Run this in the Supabase SQL Editor
-- This will set up a pg_cron job to automatically increase the 'age' of crops by 1 every day at 8:00 AM.
-- Adjust the timezone or schedule if needed. The cron string '0 8 * * *' means 8:00 AM UTC. 
-- For 8:00 AM IST, you could use '30 2 * * *' (since IST is UTC+5:30)

-- First, ensure the pg_cron extension is enabled. (Can be done from the Supabase dashboard -> Database -> Extensions -> pg_cron)

SELECT cron.schedule(
    'update_crop_age_daily', -- Job name
    '30 2 * * *',            -- Schedule (At 08:00 AM IST which is 02:30 AM UTC)
    $$
        UPDATE crop_system
        SET age = age + 1
        WHERE current_stage != 'Harvested'; -- Optional: Don't increase age if harvested
    $$
);


-- -----------------------------------------------------------------------------
-- AUTOMATED WEATHER UPDATES
-- Triggered every day at 6:00 AM and 6:00 PM IST (00:30 and 12:30 UTC)
-- This will hit the background API endpoint to update the `crop_weather` table
-- Ensure that pg_net extension is enabled alongside pg_cron.
-- -----------------------------------------------------------------------------

SELECT cron.schedule(
    'update_crop_weather_hourly',
    '0 * * * *',
    $$
        SELECT net.http_post(
            url := 'https://dharaveda.onrender.com/api/cron/update_crop_weather',
            headers := '{"Content-Type": "application/json"}'::jsonb,
            timeout_milliseconds := 5000
        );
    $$
);

-- The evening schedule is removed since it's now hourly.

-- -----------------------------------------------------------------------------
-- AUTOMATED CROP ALERTS & CONDITION UPDATES
-- Triggered every day at 6:00 PM IST (12:30 UTC)
-- This will hit the background API endpoint to update the `crop_alerts` table
-- and update `stress_level`, `health_score` in `crop_condition_snapshot` 
-- and `health_score`, `growth_progress` in `crop_system`
-- -----------------------------------------------------------------------------

SELECT cron.schedule(
    'process_daily_crop_alerts',
    '0 * * * *',
    $$
        SELECT net.http_post(
            url := 'https://dharaveda.onrender.com/api/cron/process_daily_crop_alerts',
            headers := '{"Content-Type": "application/json"}'::jsonb,
            timeout_milliseconds := 5000
        );
    $$
);
